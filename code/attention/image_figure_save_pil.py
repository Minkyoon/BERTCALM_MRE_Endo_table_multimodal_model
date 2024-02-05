from PIL import Image
import os

def create_image_grid(image_dir, output_file):
    # 폴더에서 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    image_files.sort() # 이미지 파일을 원하는 순서대로 정렬

    # 이미지 개수 확인
    num_images = len(image_files)
    num_rows = num_images // 2

    # 이미지 크기 확인 (모든 이미지가 동일한 크기라고 가정)
    sample_img = Image.open(os.path.join(image_dir, image_files[0]))
    img_width, img_height = sample_img.size

    # 새로운 이미지 그리드의 크기 계산
    grid_width = img_width * 2
    grid_height = img_height * num_rows

    # 새로운 이미지 그리드 생성
    grid_img = Image.new('RGB', (grid_width, grid_height))

    # 각 이미지를 해당 위치에 병합
    for i, image_file in enumerate(image_files):
        row = i // 2
        col = i % 2
        img = Image.open(os.path.join(image_dir, image_file))
        x_offset = col * img_width
        y_offset = row * img_height
        grid_img.paste(img, (x_offset, y_offset))

    # 그림 저장
    grid_img.save(output_file)

# 첫 번째 이미지 그리드 생성
create_image_grid("/home/minkyoon/crohn/for_clam/attention/image/tp_gradcam/", "tp_gradcam.png")

# 두 번째 이미지 그리드 생성
create_image_grid("/home/minkyoon/crohn/for_clam/attention/image/tn_gradcam/", "tn_gradcam.png")
