from src.utils import read_image


def test_read_image_not_found():
    img = read_image('not_exist.png')
    assert img is None
