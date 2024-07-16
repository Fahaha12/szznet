import os
import numpy as np
import SimpleITK as sitk
import cv2
import xml.etree.ElementTree as ET
import csv

def resize_image(slice_uint8, target_size=(512, 512)):
    """提高图像分辨率"""
    # 插值提高分辨率
    resized_img = cv2.resize(slice_uint8, target_size, interpolation=cv2.INTER_CUBIC)
    return resized_img

import xml.etree.ElementTree as ET

def categorize_age(age):
    """将年龄从0-100分为5段"""
    try:
        age = int(float(age))  # 确保 `age` 是整数 确保 `age` 是整数
        if age < 20:
            return "1"
        elif age < 40:
            return "2"
        elif age < 60:
            return "3"
        elif age < 80:
            return "4"
        else:
            return "5"
    except ValueError:
        return "5"        

def categorize_survival_days(survival_days):
    """将生存天数分为5级，如果出现非数字，则显示5"""
    try:
        survival_days = int(float(survival_days))  # 确保 `survival_days` 是整数
        if survival_days < 200:
            return "1"
        elif survival_days < 400:
            return "2"
        elif survival_days < 600:
            return "3"
        elif survival_days < 800:
            return "4"
        else:
            return "5"
    except ValueError:
        return "5"

def categorize_extent_of_resection(extent_of_resection):
    """将Extent_of_Resection转换为1或2"""
    return "1" if extent_of_resection == "GTR" else "2"

def create_annotation(brats20_id, age, survival_days, extent_of_resection, modality_name, slice_idx, bbox, output_path):
    """创建XML注释文件"""
    # 创建XML的根元素
    annotation = ET.Element("annotation")
    
    # 创建子元素
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "BraTS2020"

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = "512"
    height = ET.SubElement(size, "height")
    height.text = "512"
    depth = ET.SubElement(size, "depth")
    depth.text = "3"

    entity = ET.SubElement(annotation, "Entity")
    entity.text = "Brain Tumor"

    imageZposition = ET.SubElement(annotation, "imageZposition")
    imageZposition.text = str(slice_idx)

    imageSOP_UID = ET.SubElement(annotation, "imageSOP_UID")
    imageSOP_UID.text = f"{brats20_id}_{modality_name}_{slice_idx}"

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    object_elem = ET.SubElement(annotation, "object")
    name_elem = ET.SubElement(object_elem, "name")
    name_elem.text = "Brain Tumor"

    pose = ET.SubElement(object_elem, "pose")
    pose.text = "Unspecified"

    truncated = ET.SubElement(object_elem, "truncated")
    truncated.text = "0"

    difficult = ET.SubElement(object_elem, "difficult")
    difficult.text = "0"

    noduleID = ET.SubElement(object_elem, "noduleID")
    noduleID.text = "Nodule 001"

    characteristics = ET.SubElement(object_elem, "characteristics")
    age_elem = ET.SubElement(characteristics, "Age")
    age_elem.text = categorize_age(age)

    survival_days_elem = ET.SubElement(characteristics, "Survival_days")
    survival_days_elem.text = categorize_survival_days(survival_days)

    extent_of_resection_elem = ET.SubElement(characteristics, "Extent_of_Resection")
    extent_of_resection_elem.text = categorize_extent_of_resection(extent_of_resection)

    calcification = ET.SubElement(characteristics, "calcification")
    calcification.text = "0"

    sphericity = ET.SubElement(characteristics, "sphericity")
    sphericity.text = "0"

    margin = ET.SubElement(characteristics, "margin")
    margin.text = "0"

    lobulation = ET.SubElement(characteristics, "lobulation")
    lobulation.text = "0"

    spiculation = ET.SubElement(characteristics, "spiculation")
    spiculation.text = "0"

    texture = ET.SubElement(characteristics, "texture")
    texture.text = "0"

    inclusion = ET.SubElement(object_elem, "inclusion")
    inclusion.text = "TRUE"

    bndbox = ET.SubElement(object_elem, "bndbox")
    xmin_elem = ET.SubElement(bndbox, "xmin")
    xmin_elem.text = str(bbox[0])
    ymin_elem = ET.SubElement(bndbox, "ymin")
    ymin_elem.text = str(bbox[1])
    xmax_elem = ET.SubElement(bndbox, "xmax")
    xmax_elem.text = str(bbox[2])
    ymax_elem = ET.SubElement(bndbox, "ymax")
    ymax_elem.text = str(bbox[3])

    # 将XML树保存到文件
    tree = ET.ElementTree(annotation)
    xml_output_path = output_path.replace(".jpg", ".xml")
    tree.write(xml_output_path, encoding="utf-8", xml_declaration=True)

def find_bounding_box(slice_seg):
    """从分割图像中找到标注框"""
    y_indices, x_indices = np.where(slice_seg > 0)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return [0, 0, 0, 0]  # 如果没有找到标注框，则返回空框
    xmin, xmax = x_indices.min(), x_indices.max()
    ymin, ymax = y_indices.min(), y_indices.max()
    return [xmin, ymin, xmax, ymax]

def process_mri_and_segmentation(mri_path, seg_path, output_dir, modality_name, brats20_id, age, survival_days, extent_of_resection, num_slices=6, threshold=0.1, target_size=(512, 512)):
    """处理MRI和分割文件并生成JPG和XML"""
    # 读取MRI和分割文件
    mri_img = sitk.ReadImage(mri_path)
    seg_img = sitk.ReadImage(seg_path)

    # 将SimpleITK图像转换为numpy数组
    mri_array = sitk.GetArrayFromImage(mri_img)
    seg_array = sitk.GetArrayFromImage(seg_img)

    # 确保MRI和分割图像具有相同的形状
    assert mri_array.shape == seg_array.shape, "MRI and segmentation shapes do not match"

    # 找出肿瘤特征明显的切片
    slices_with_tumor = []
    for i in range(mri_array.shape[0]):
        slice_seg = seg_array[i]
        tumor_ratio = np.sum(slice_seg) / slice_seg.size
        slices_with_tumor.append((i, tumor_ratio))

    # 按照肿瘤占比排序并选择前 num_slices 张
    slices_with_tumor = sorted(slices_with_tumor, key=lambda x: x[1], reverse=True)[:num_slices]

    # 如果切片数量不足，补齐
    if len(slices_with_tumor) < num_slices and slices_with_tumor:
        slices_with_tumor = slices_with_tumor + [slices_with_tumor[-1]] * (num_slices - len(slices_with_tumor))
    elif not slices_with_tumor:
        print(f"No significant tumor found in modality: {modality_name}")
        return

    # 保存选定的切片
    for idx, (slice_idx, _) in enumerate(slices_with_tumor):
        slice_mri = mri_array[slice_idx]
        slice_seg = seg_array[slice_idx]

        # 标注框
        bbox = find_bounding_box(slice_seg)

        # 正常化并转换为uint8
        slice_normalized = cv2.normalize(slice_mri, None, 0, 255, cv2.NORM_MINMAX)
        slice_uint8 = np.uint8(slice_normalized)

        # 提高分辨率
        resized_img = resize_image(slice_uint8, target_size)

        # 获取文件夹中的三位数字标识
        folder_id = os.path.basename(output_dir).split("_")[-1]
        identifier = f"{int(folder_id):03d}"

        # 保存为灰度JPG，设置更高的JPEG质量
        output_path = os.path.join(output_dir, f"BraTS20_Training_{identifier}_{modality_name}_{idx:03d}.jpg")
        cv2.imwrite(output_path, resized_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

        # 创建并保存对应的XML文件
        create_annotation(brats20_id, age, survival_days, extent_of_resection, modality_name, slice_idx, bbox, output_path)

def process_case(case_dir, output_base_dir, csv_data, num_slices=6, threshold=0.1, target_size=(512, 512)):
    """处理单个病例"""
    # 获取病例文件夹名称
    case_name = os.path.basename(case_dir)
    output_dir = os.path.join(output_base_dir, case_name)
    os.makedirs(output_dir, exist_ok=True)

    # 找到所有MRI模态和分割文件
    modalities = ["flair", "t1", "t1ce", "t2"]
    mri_files = {mod: os.path.join(case_dir, f"{case_name}_{mod}.nii") for mod in modalities}
    seg_file = os.path.join(case_dir, f"{case_name}_seg.nii")

    if not os.path.exists(seg_file):
        print(f"Segmentation file not found for case {case_name}")
        return

    # 从CSV数据中获取病人的相关信息
    patient_info = csv_data.get(case_name, {})
    age = patient_info.get('Age', 'NA')
    survival_days = patient_info.get('Survival_days', 'NA')
    extent_of_resection = patient_info.get('Extent_of_Resection', 'NA')

    # 处理每个模态
    for modality, mri_path in mri_files.items():
        if os.path.exists(mri_path):
            print(f"Processing {case_name} - {modality}")
            process_mri_and_segmentation(mri_path, seg_file, output_dir, modality, case_name, age, survival_days, extent_of_resection, num_slices, threshold, target_size)
        else:
            print(f"MRI file not found: {mri_path}")

def load_csv_data(csv_path):
    """加载CSV数据"""
    csv_data = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            brats20_id = row['Brats20ID']
            csv_data[brats20_id] = {
                'Age': row['Age'],
                'Survival_days': row['Survival_days'],
                'Extent_of_Resection': row['Extent_of_Resection']
            }
    return csv_data

def process_all_cases(base_dir, output_base_dir, csv_path, num_slices=6, threshold=0.1, target_size=(512, 512)):
    """处理所有病例"""
    # 获取所有病例文件夹
    case_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # 加载CSV数据
    csv_data = load_csv_data(csv_path)

    # 处理每个病例
    for case_dir in case_dirs:
        process_case(case_dir, output_base_dir, csv_data, num_slices, threshold, target_size)

# 使用示例
base_directory = r'C:\Users\suzuk\Downloads\BraTS2020_TrainingData-Survivial\MICCAI_BraTS2020_TrainingData'
output_directory = r'C:\Users\suzuk\Downloads\BraTS2020_TrainingData-Survivial\jpg'
csv_path = r'C:\Users\suzuk\Downloads\BraTS2020_TrainingData-Survivial\MICCAI_BraTS2020_TrainingData\survival_info.csv'
num_slices = 6  # 每个模态保存的切片数量
threshold_value = 0.1  # 调整此值以控制肿瘤区域的比例阈值
target_image_size = (512, 512)  # 目标分辨率

process_all_cases(base_directory, output_directory, csv_path, num_slices, threshold_value, target_image_size)