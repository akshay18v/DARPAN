import cv2
import os


def preprocess_image(image_path, save_path=None):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, thresh)

    return img, thresh


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")

    print("Starting preprocessing...")

    for platform in os.listdir(raw_dir):

        platform_path = os.path.join(raw_dir, platform)

        if not os.path.isdir(platform_path):
            continue

        output_platform_dir = os.path.join(processed_dir, platform)
        os.makedirs(output_platform_dir, exist_ok=True)

        print(f"\nProcessing {platform}...")

        for img_name in os.listdir(platform_path):

            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):

                img_path = os.path.join(platform_path, img_name)

                base_name = os.path.splitext(img_name)[0]

                save_path = os.path.join(
                    output_platform_dir,
                    f"{base_name}_processed.png"
                )

                try:
                    preprocess_image(img_path, save_path)
                    print(f"Processed: {img_name}")

                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

    print("\nAll images preprocessed successfully!")