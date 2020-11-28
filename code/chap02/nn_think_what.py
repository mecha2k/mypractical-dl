from code.utils.kerasutils import init_gpus
from code.chap02.visualization import process_image

import os


def main():
    img_path = "../../images/sample-images/dog.jpg"
    output_prefix = os.path.splitext(os.path.basename(img_path))[0]
    process_image(img_path, output_prefix + "_output.jpg")


if __name__ == "__main__":
    init_gpus()
    main()
