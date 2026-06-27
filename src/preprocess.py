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

    test_image = os.path.join(
        base_dir,
        "data",
        "raw",
        "airbnb",
        "airbnb_3.png"
    )

    processed_image_path = os.path.join(
        base_dir,
        "data",
        "processed",
        "airbnb",
        "airbnb_0_processed.png"
    )

    try:
        original, processed = preprocess_image(
            test_image,
            processed_image_path
        )

        print("Image preprocessing successful!")
        print(f"Saved to: {processed_image_path}")

        cv2.imshow("Original Image", original)
        cv2.imshow("Preprocessed Image", processed)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")