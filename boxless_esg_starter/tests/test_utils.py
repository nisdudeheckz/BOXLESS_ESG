from boxless_esg.utils import Box, iou_boxes

def test_iou_identity():
    a = Box(0,0,10,10); b = Box(0,0,10,10)
    assert abs(iou_boxes(a,b) - 1.0) < 1e-6
