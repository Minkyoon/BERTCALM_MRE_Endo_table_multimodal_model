import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def create_image_grid(image_dir, output_file):
    # 폴더에서 이미지 파일 목록 가져오기
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    image_files.sort() # 이미지 파일을 원하는 순서대로 정렬

    # 이미지 개수 확인
    num_images = len(image_files)
    num_rows = num_images // 2

    # 그림 생성
    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 5*num_rows))

    # 각 이미지를 해당 위치에 표시
    for i, image_file in enumerate(image_files):
        row = i // 2
        col = i % 2
        img = mpimg.imread(os.path.join(image_dir, image_file))
        axs[row, col].imshow(img)
        axs[row, col].axis('off') # 축 제거

    # 이미지 간의 간격 제거
    plt.subplots_adjust(wspace=0, hspace=0)

    # 그림 저장
    fig.savefig(output_file, bbox_inches='tight', pad_inches=0)

    # 그림 표시 (필요한 경우)
    plt.show()

# 첫 번째 이미지 그리드 생성
create_image_grid("/home/minkyoon/crohn/for_clam/attention/image/tp/", "tp.png")

# 두 번째 이미지 그리드 생성
create_image_grid("/home/minkyoon/crohn/for_clam/attention/image/tn/", "tn.png")
