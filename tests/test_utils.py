from src.utils import read_image


def test_read_image_not_found():
    try:
        read_image('not_exist.png')
    except Exception:
        assert True
    else:
        assert False
