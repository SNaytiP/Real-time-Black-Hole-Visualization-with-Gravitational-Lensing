import numpy as np
from PIL import Image

def generate_starfield(width=1280, height=720, n_stars=3000, seed=42):
    np.random.seed(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # ดาวธรรมดา
    xs = np.random.randint(0, width,  n_stars)
    ys = np.random.randint(0, height, n_stars)
    brightness = np.random.randint(100, 255, n_stars)

    for x, y, b in zip(xs, ys, brightness):
        img[y, x] = [b, b, b]

    # ดาวสว่าง (มี glow)
    for _ in range(80):
        x  = np.random.randint(10, width  - 10)
        y  = np.random.randint(10, height - 10)
        b  = np.random.randint(180, 255)
        r  = np.random.choice([3, 4, 5])

        # กระจาย glow รอบดาวสว่าง
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                dist = np.sqrt(dx**2 + dy**2)
                if dist <= r:
                    glow = int(b * np.exp(-dist**2 / (r**2 * 0.5)))
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        img[ny, nx] = np.minimum(
                            img[ny, nx] + glow, 255
                        )

    # nebula — noise ฝ้าสีฟ้า-ม่วง
    from PIL import Image, ImageFilter
    pil = Image.fromarray(img)
    nebula = np.random.rand(height // 8, width // 8) * 40
    nebula_img = Image.fromarray(nebula.astype(np.uint8)).resize(
        (width, height), Image.BILINEAR
    ).filter(ImageFilter.GaussianBlur(radius=12))
    nebula_arr = np.array(nebula_img)

    # ใส่สี nebula (ฟ้า-ม่วง)
    result = np.array(pil, dtype=np.float32)
    result[:, :, 2] += nebula_arr * 1.2   # blue channel
    result[:, :, 0] += nebula_arr * 0.4   # red channel
    result = np.clip(result, 0, 255).astype(np.uint8)

    return Image.fromarray(result)

if __name__ == "__main__":
    img = generate_starfield()
    img.save("starfield.png")
    print("starfield.png saved!")