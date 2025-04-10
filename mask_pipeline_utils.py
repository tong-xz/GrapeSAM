import torch
import numpy as np
from typing import Optional, Tuple
from model.utils import ResizeLongestSide
from segment_anything.predictor import SamPredictor
from nifty.tools import blocking
from skimage.segmentation import find_boundaries
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt


def _compute_logits_from_mask(mask, eps=1e-3):

    def inv_sigmoid(x):
        return np.log(x / (1 - x))

    logits = np.zeros(mask.shape, dtype="float32")
    logits[mask == 1] = 1 - eps
    logits[mask == 0] = eps
    logits = inv_sigmoid(logits)

    # resize to the expected mask shape of SAM (256x256)
    assert logits.ndim == 2
    expected_shape = (256, 256)

    if logits.shape == expected_shape:  # shape matches, do nothing
        pass

    elif logits.shape[0] == logits.shape[1]:  # shape is square
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image(logits[..., None])

    else:  # shape is not square
        # resize the longest side to expected shape
        trafo = ResizeLongestSide(expected_shape[0])
        logits = trafo.apply_image(logits[..., None])

        # pad the other side
        h, w = logits.shape
        padh = expected_shape[0] - h
        padw = expected_shape[1] - w
        # IMPORTANT: need to pad with zero, otherwise SAM doesn't understand the padding
        pad_width = ((0, padh), (0, padw))
        logits = np.pad(logits, pad_width, mode="constant", constant_values=0)

    logits = logits[None]
    assert logits.shape == (1, 256, 256), f"{logits.shape}"
    return logits


def segment_from_mask(
    predictor: SamPredictor,
    mask: np.ndarray,
    image_embeddings=None,
    i: Optional[int] = None,
    use_box: bool = True,
    use_mask: bool = True,
    use_points: bool = False,
    original_size: Optional[Tuple[int, ...]] = None,
    multimask_output: bool = False,
    return_all: bool = False,
    return_logits: bool = False,
    box_extension: float = 0.0,
    box: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    use_single_point: bool = False,
):
    """Segmentation from a mask prompt.

    Args:
        predictor: The segment anything predictor.
        mask: The mask used to derive prompts.
        image_embeddings: Optional precomputed image embeddings.
            Has to be passed if the predictor is not yet initialized.
         i: Index for the image data. Required if the input data has three spatial dimensions
             or a time dimension and two spatial dimensions.
        use_box: Whether to derive the bounding box prompt from the mask.
        use_mask: Whether to use the mask itself as prompt.
        use_points: Whether to derive point prompts from the mask.
        original_size: Full image shape. Use this if the mask that is being passed
            downsampled compared to the original image.
        multimask_output: Whether to return multiple or just a single mask.
        return_all: Whether to return the score and logits in addition to the mask.
        box_extension: Relative factor used to enlarge the bounding box prompt.
        box: Precomputed bounding box.
        points: Precomputed point prompts.
        labels: Positive/negative labels corresponding to the point prompts.
        use_single_point: Whether to derive just a single point from the mask.
            In case use_points is true.

    Returns:
        The binary segmentation mask.
    """
    prompts = (mask, box, points, labels)

    def _to_tile(prompts, shape, tile_shape, halo):
        mask, box, points, labels = prompts
        tile_id, tile, mask = _mask_to_tile(mask, shape, tile_shape, halo)
        if points is not None:
            tile_id_points, tile, point_prompts = _points_to_tile(
                (points, labels), shape, tile_shape, halo
            )
            if tile_id_points != tile_id:
                raise RuntimeError(
                    f"Inconsistent tile ids for mask and point prompts: {tile_id_points} != {tile_id}."
                )
            points, labels = point_prompts
        if box is not None:
            tile_id_box, tile, box = _box_to_tile(box, shape, tile_shape, halo)
            if tile_id_box != tile_id:
                raise RuntimeError(
                    f"Inconsistent tile ids for mask and box prompts: {tile_id_box} != {tile_id}."
                )
        return tile_id, tile, (mask, box, points, labels)

    predictor, tile, prompts, shape = _initialize_predictor(
        predictor, image_embeddings, i, prompts, _to_tile
    )
    mask, box, points, labels = prompts

    logits = _compute_logits_from_mask(mask) if use_mask else None

    mask, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=logits,
        box=box,
        multimask_output=multimask_output,
        return_logits=return_logits,
    )

    if tile is not None:
        mask = _tile_to_full_mask(mask, shape, tile)

    if return_all:
        return mask, scores, logits
    else:
        return mask


# compute the bounding box from a mask. SAM expects the following input:
# box (np.ndarray or None): A length 4 array given a box prompt to the model, in XYXY format.
def _compute_box_from_mask(mask, original_size=None, box_extension=0):
    coords = np.where(mask == 1)
    min_y, min_x = coords[0].min(), coords[1].min()
    max_y, max_x = coords[0].max(), coords[1].max()
    box = np.array([min_y, min_x, max_y + 1, max_x + 1])
    return _process_box(
        box, mask.shape, original_size=original_size, box_extension=box_extension
    )


def _process_box(box, shape, original_size=None, box_extension=0):
    if box_extension == 0:  # no extension
        extension_y, extension_x = 0, 0
    elif box_extension >= 1:  # extension by a fixed factor
        extension_y, extension_x = box_extension, box_extension
    else:  # extension by fraction of the box len
        len_y, len_x = box[2] - box[0], box[3] - box[1]
        extension_y, extension_x = box_extension * len_y, box_extension * len_x

    box = np.array(
        [
            max(box[1] - extension_x, 0),
            max(box[0] - extension_y, 0),
            min(box[3] + extension_x, shape[1]),
            min(box[2] + extension_y, shape[0]),
        ]
    )

    if original_size is not None:
        trafo = ResizeLongestSide(max(original_size))
        box = trafo.apply_boxes(box[None], (256, 256)).squeeze()

    # round up the bounding box values
    box = np.round(box).astype(int)

    return box


def _box_to_tile(box, shape, tile_shape, halo):
    tiling = blocking([0, 0], shape, tile_shape)
    center = (
        np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        .round()
        .astype("int")
        .tolist()
    )
    tile_id = tiling.coordinatesToBlockId(center)

    tile = tiling.getBlockWithHalo(tile_id, list(halo)).outerBlock
    offset = tile.begin
    this_tile_shape = tile.shape

    box_in_tile = np.array(
        [
            max(box[0] - offset[0], 0),
            max(box[1] - offset[1], 0),
            min(box[2] - offset[0], this_tile_shape[0]),
            min(box[3] - offset[1], this_tile_shape[1]),
        ]
    )

    return tile_id, tile, box_in_tile


def _mask_to_tile(mask, shape, tile_shape, halo):
    tiling = blocking([0, 0], shape, tile_shape)

    coords = np.where(mask)
    center = (
        np.array([np.mean(coords[0]), np.mean(coords[1])])
        .round()
        .astype("int")
        .tolist()
    )
    tile_id = tiling.coordinatesToBlockId(center)

    tile = tiling.getBlockWithHalo(tile_id, list(halo)).outerBlock
    bb = tuple(slice(beg, end) for beg, end in zip(tile.begin, tile.end))

    mask_in_tile = mask[bb]
    return tile_id, tile, mask_in_tile


def _initialize_predictor(predictor, image_embeddings, i, prompts, to_tile):
    tile = None

    # Set the precomputed state for tiled prediction.
    if image_embeddings is not None and image_embeddings["input_size"] is None:
        features = image_embeddings["features"]
        shape, tile_shape, halo = (
            features.attrs["shape"],
            features.attrs["tile_shape"],
            features.attrs["halo"],
        )
        tile_id, tile, prompts = to_tile(prompts, shape, tile_shape, halo)
        set_precomputed(predictor, image_embeddings, i, tile_id=tile_id)

    # Set the precomputed state for normal prediction.
    elif image_embeddings is not None:
        shape = image_embeddings["original_size"]
        set_precomputed(predictor, image_embeddings, i)

    else:
        shape = predictor.original_size

    return predictor, tile, prompts, shape


def _points_to_tile(prompts, shape, tile_shape, halo):
    points, labels = prompts

    tiling = blocking([0, 0], shape, tile_shape)
    center = np.mean(points, axis=0).round().astype("int").tolist()
    tile_id = tiling.coordinatesToBlockId(center)

    tile = tiling.getBlockWithHalo(tile_id, list(halo)).outerBlock
    offset = tile.begin
    this_tile_shape = tile.shape

    points_in_tile = points - np.array(offset)
    labels_in_tile = labels

    valid_point_mask = (points_in_tile >= 0).all(axis=1)
    valid_point_mask = np.logical_and(
        valid_point_mask,
        np.logical_and(
            points_in_tile[:, 0] < this_tile_shape[0],
            points_in_tile[:, 1] < this_tile_shape[1],
        ),
    )
    if not valid_point_mask.all():
        points_in_tile = points_in_tile[valid_point_mask]
        labels_in_tile = labels_in_tile[valid_point_mask]
        print(
            f"{(~valid_point_mask).sum()} points were not in the tile and are dropped"
        )

    return tile_id, tile, (points_in_tile, labels_in_tile)

import numpy as np

# Assume _compute_box_from_mask is already defined elsewhere
# For example, it might compute the minimal box containing the object,
# and then expand it by box_extension.

def _compute_points_from_mask_fixed_points(
    mask, original_size, box_extension, num_points=10
):
    """
    Generate a fixed number of point prompts from a binary mask.
    
    This method first computes a bounding box from the mask, then crops the mask
    to that region. It randomly selects points from inside the mask (positive points)
    and from outside the mask (negative points), with the total number of points controlled 
    by the `num_points` parameter.
    
    Parameters:
        mask (ndarray): A binary mask where nonzero values represent the object.
        original_size (tuple or None): The (height, width) of the original image.
            If provided, the computed coordinates are scaled to the original image size.
        box_extension (int or float): A value to expand the bounding box around the mask.
        num_points (int): Total number of points to generate.
        
    Returns:
        point_coords (ndarray): An array of shape (num_points, 2) containing (x, y) coordinates.
        point_labels (ndarray): A 1D array of length num_points with labels
            (1 for positive inside the mask, 0 for negative outside the mask).
    """
    
    # 1. Compute the bounding box of the mask (using your existing helper)
    box = _compute_box_from_mask(mask, box_extension=box_extension)
    # The box is assumed to be in the format [min_x, min_y, max_x, max_y]
    # We convert it to Python slices (note the swap: rows correspond to y, cols to x)
    bb = (slice(box[1], box[3]), slice(box[0], box[2]))
    offset = np.array([box[1], box[0]])  # offset to convert cropped coords to full image coords

    # 2. Crop the mask to the bounding box and ensure it is boolean
    cropped_mask = mask[bb].astype(bool)
    
    # 3. Find all pixel coordinates inside and outside the mask
    #    np.argwhere returns coordinates in (row, col) order (i.e. y, x)
    pos_indices = np.argwhere(cropped_mask)      # positive points (inside the mask)
    neg_indices = np.argwhere(~cropped_mask)     # negative points (outside the mask)
    
    # 4. Decide how many positive and negative points to sample.
    #    Here we choose to assign roughly half of the total points to positives.
    num_pos = num_points // 2 + num_points % 2  # e.g., if num_points is odd, positives get the extra point
    num_neg = num_points // 2
    
    # 5. Sample points (with replacement if necessary)
    if len(pos_indices) > 0:
        if len(pos_indices) >= num_pos:
            chosen_pos = pos_indices[np.random.choice(len(pos_indices), num_pos, replace=False)]
        else:
            # Not enough positive pixels: sample with replacement to reach the desired number
            chosen_pos = pos_indices[np.random.choice(len(pos_indices), num_pos, replace=True)]
    else:
        chosen_pos = np.empty((0, 2), dtype=int)
    
    if len(neg_indices) > 0:
        if len(neg_indices) >= num_neg:
            chosen_neg = neg_indices[np.random.choice(len(neg_indices), num_neg, replace=False)]
        else:
            chosen_neg = neg_indices[np.random.choice(len(neg_indices), num_neg, replace=True)]
    else:
        chosen_neg = np.empty((0, 2), dtype=int)
    
    # 6. Combine the selected points and add the offset back to get coordinates in the full mask
    point_coords = np.concatenate([chosen_pos, chosen_neg], axis=0).astype("float64")
    point_coords += offset  # adjust the coordinates to the original mask
    
    # 7. If an original size is provided, rescale the coordinates accordingly.
    if original_size is not None:
        # original_size is expected to be (height, width)
        scale_factor = np.array([
            original_size[0] / float(mask.shape[0]),
            original_size[1] / float(mask.shape[1])
        ])
        point_coords *= scale_factor
    
    # 8. Create the corresponding point labels (1 for inside, 0 for outside)
    point_labels = np.concatenate([
        np.ones(len(chosen_pos), dtype="uint8"),
        np.zeros(len(chosen_neg), dtype="uint8")
    ])
    
    # 9. SAM expects the coordinates in (x, y) order.
    #    Currently, point_coords are in (row, col) i.e. (y, x) order.
    #    So we swap the two columns.
    return point_coords[:, ::-1], point_labels

# sample points from a mask. SAM expects the following point inputs:
def _compute_points_from_mask(
    mask, original_size, box_extension, use_single_point=False
):
    box = _compute_box_from_mask(mask, box_extension=box_extension)

    # get slice and offset in python coordinate convention
    bb = (slice(box[1], box[3]), slice(box[0], box[2]))
    offset = np.array([box[1], box[0]])

    # crop the mask and compute distances
    cropped_mask = mask[bb]
    object_boundaries = find_boundaries(cropped_mask, mode="outer")
    distances = gaussian(distance_transform_edt(object_boundaries == 0))
    inner_distances = distances.copy()
    cropped_mask = cropped_mask.astype("bool")
    inner_distances[~cropped_mask] = 0.0
    if use_single_point:
        center = inner_distances.argmax()
        center = np.unravel_index(center, inner_distances.shape)
        point_coords = (center + offset)[None]
        point_labels = np.ones(1, dtype="uint8")
        return point_coords[:, ::-1], point_labels

    outer_distances = distances.copy()
    outer_distances[cropped_mask] = 0.0

    # sample positives and negatives from the distance maxima
    inner_maxima = peak_local_max(inner_distances, exclude_border=False, min_distance=3)
    outer_maxima = peak_local_max(outer_distances, exclude_border=False, min_distance=5)

    # derive the positive (=inner maxima) and negative (=outer maxima) points
    point_coords = np.concatenate([inner_maxima, outer_maxima]).astype("float64")
    point_coords += offset

    if original_size is not None:
        scale_factor = np.array(
            [
                original_size[0] / float(mask.shape[0]),
                original_size[1] / float(mask.shape[1]),
            ]
        )[None]
        point_coords *= scale_factor

    # get the point labels
    point_labels = np.concatenate(
        [
            np.ones(len(inner_maxima), dtype="uint8"),
            np.zeros(len(outer_maxima), dtype="uint8"),
        ]
    )
    return point_coords[:, ::-1], point_labels


def set_precomputed(
    predictor: SamPredictor,
    image_embeddings,
    i: Optional[int] = None,
    tile_id: Optional[int] = None,
) -> SamPredictor:
    """Set the precomputed image embeddings for a predictor.

    Args:
        predictor: The SegmentAnything predictor.
        image_embeddings: The precomputed image embeddings computed by `precompute_image_embeddings`.
        i: Index for the image data. Required if `image` has three spatial dimensions
            or a time dimension and two spatial dimensions.
        tile_id: Index for the tile. This is required if the embeddings are tiled.

    Returns:
        The predictor with set features.
    """
    if tile_id is not None:
        tile_features = image_embeddings["features"][tile_id]
        tile_image_embeddings = {
            "features": tile_features,
            "input_size": tile_features.attrs["input_size"],
            "original_size": tile_features.attrs["original_size"],
        }
        return set_precomputed(predictor, tile_image_embeddings, i=i)

    device = predictor.device
    features = image_embeddings["features"]
    assert features.ndim in (4, 5), f"{features.ndim}"
    if features.ndim == 5 and i is None:
        raise ValueError("The data is 3D so an index i is needed.")
    elif features.ndim == 4 and i is not None:
        raise ValueError("The data is 2D so an index is not needed.")

    if i is None:
        predictor.features = (
            features.to(device)
            if torch.is_tensor(features)
            else torch.from_numpy(features[:]).to(device)
        )
    else:
        predictor.features = (
            features[i].to(device)
            if torch.is_tensor(features)
            else torch.from_numpy(features[i]).to(device)
        )
    predictor.original_size = image_embeddings["original_size"]
    predictor.input_size = image_embeddings["input_size"]
    predictor.is_image_set = True

    return predictor
