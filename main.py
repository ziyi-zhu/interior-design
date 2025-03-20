from designer import DesignConfig, Designer

if __name__ == "__main__":
    config = DesignConfig(
        input_dir="inputs",
        output_dir="outputs/lavender_reseda",
        transform_type="canny",
        prompt=(
            "interior design of a bright and cozy apartment with warm beige walls, "
            "polished reddish-brown hardwood floors and soft natural light, "
            # "featuring a harmonious blend of terracotta, olive green, and natural wood tones"
            # "featuring a harmonious blend of navy blue, burnt orange, and natural wood tones"
            # "featuring a harmonious blend of dusty rose, deep burgundy, and natural wood tones"
            "featuring a harmonious blend of lavender blush, reseda green, and natural wood tones"
        ),
        negative_prompt=(
            "(hands), text, error, cropped, (worst quality:1.2), (low quality:1.2), "
            "normal quality, (jpeg artifacts:1.3), signature, watermark, username, "
            "blurry, artist name, monochrome, sketch, censorship, censor, (copyright:1.2)"
        ),
    )
    designer = Designer(config)
    designer.design()
