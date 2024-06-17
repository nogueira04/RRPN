from detectron2.structures import Boxes

def clip_boxes_to_image(boxes, image_shape):
    """
    boxes: tensor of shape (N, 4), where N is the number of boxes, and each row contains (x1, y1, x2, y2).
    image_shape: tuple, the shape of the image in the format (height, width).
    """
    boxes = Boxes(boxes)
    boxes.clip(image_shape)
    return boxes.tensor
