import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans


for i in range(1, 7):

    img = Image.imread(f"imgs\\test_{i}.jpg")

 
    print(img.shape)



    if i != 4:
        img = img.astype(np.float64) / 255

    w, h, d = img.shape
    img_array = np.reshape(img, (w * h, d))

    img_array_aprox = img_array.copy()

    n_colors = 5  
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans = kmeans.fit(img_array)
    labels = kmeans.predict(img_array)

    img_array_aprox = kmeans.cluster_centers_[labels]

    quantized_img = img_array_aprox.reshape(w, h, d)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Kvantizirana slika ({n_colors} boja)")
    plt.imshow(quantized_img)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

 

    inertia_values = []

    k_range = range(1, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(img_array)
        inertia_values.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(k_range, inertia_values, marker='o', color='b')
    plt.title("Ovisnost inercije o broju grupa K")
    plt.xlabel("Broj grupa (K)")
    plt.ylabel("Inercija")
    plt.grid()
    plt.show()


for i in range(1, 7):
    img = Image.imread(f"imgs\\test_{i}.jpg")

    if i != 4:
        img = img.astype(np.float64) / 255

    w, h, d = img.shape
    img_array = np.reshape(img, (w * h, d))

    img_array_aprox = img_array.copy()

    n_colors = 5  
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans = kmeans.fit(img_array)
    labels = kmeans.predict(img_array)

    for k in range(n_colors):
        binary_mask = (labels == k).astype(np.uint8).reshape(w, h)
        
        plt.figure()
        plt.title(f"Binarna slika za grupu {k}")
        plt.imshow(binary_mask, cmap='gray')
        plt.axis("off")
        plt.show()
