import os
import matplotlib.pyplot as plt
import cv2

def load_image(image_name_folder_path: str):
    try:
        loaded_image = cv2.imread(image_name_folder_path)
        if loaded_image is None:
            raise ValueError("Image not found")
        return loaded_image
    except Exception as e:
        print(f"Error loading image {image_name_folder_path}. Error: {e}")
        return None

def main():
    image_folder_path = "/home/wyctor/PROJETOS/multimodal-model-skin-lesion-classifier/data/ISIC2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
    # wanted_image_list = [
    #     {"class":"BCC", "image":"PAT_46_881_939.png"}, 
    #     {"class":"ACK", "image":"PAT_705_4015_413.png"}, 
    #     {"class":"SCC", "image":"PAT_380_1540_959.png"},
    #     {"class":"SEK", "image":"PAT_107_160_609.png"}, 
    #     {"class":"NEV", "image":"PAT_793_1512_327.png"}, 
    #     {"class":"MEL", "image":"PAT_680_1289_182.png"}
    # ]
    wanted_image_list = [
        {"class":"NV", "image":"ISIC_0000000.jpg"}, 
        {"class":"MEL", "image":"ISIC_0000002.jpg"}, 
        {"class":"BKL", "image":"ISIC_0010491.jpg"},
        {"class":"VASC", "image":"ISIC_0024370.jpg"}, 
        {"class":"SCC", "image":"ISIC_0024372.jpg"}, 
        {"class":"BCC", "image": "ISIC_0024403.jpg"},
        {"class":"AK", "image": "ISIC_0024468.jpg"},
        {"class":"DF", "image":"ISIC_0024386.jpg"}
    ]

    # Create a figure for the plot
    f, axarr = plt.subplots(nrows=2, ncols=4, figsize=(12, 8))  # Adjust the grid size to match the number of images
    axarr = axarr.ravel()  # Flatten the array of axes for easy indexing

    for i, item in enumerate(wanted_image_list):
        image_class = item["class"]
        image_name = item["image"]
        img = load_image(os.path.join(image_folder_path, image_name))
        
        if img is not None:
            axarr[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for proper display in matplotlib
            axarr[i].set_title(image_class)
            axarr[i].axis('off')  # Hide axis for a cleaner image display

    plt.tight_layout()  # To adjust spacing between subplots
    plt.show()

if __name__ == "__main__":
    main()
