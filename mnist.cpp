/*  Used to load in images and their labels of whatever that weird
    format MNIST comes in.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

struct MNISTData {
    std::vector<std::vector<uint8_t>> images; // Each image is a vector of bytes
    std::vector<uint8_t> labels;
    int numImages;
    int rows;
    int cols;
};

int readInt32(std::ifstream& file) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    return (static_cast<int>(bytes[0]) << 24) |
           (static_cast<int>(bytes[1]) << 16) |
           (static_cast<int>(bytes[2]) << 8) |
           static_cast<int>(bytes[3]);
}

// Function to load MNIST images and labels
MNISTData loadMNIST(const std::string& imageFile, const std::string& labelFile) {
    MNISTData data;

    // Open the image file
    std::ifstream imgFile(imageFile, std::ios::binary);
    if (!imgFile.is_open()) {
        throw std::runtime_error("Unable to open image file");
    }

    // Read header for image file
    int magicNumber = readInt32(imgFile);
    data.numImages = readInt32(imgFile);
    data.rows = readInt32(imgFile);
    data.cols = readInt32(imgFile);

    if (magicNumber != 2051) {
        throw std::runtime_error("Invalid MNIST image file format");
    }

    // Read image data
    for (int i = 0; i < data.numImages; ++i) {
        std::vector<uint8_t> image(data.rows * data.cols);
        imgFile.read(reinterpret_cast<char*>(image.data()), data.rows * data.cols);
        data.images.push_back(image);
    }

    // Open the label file
    std::ifstream lblFile(labelFile, std::ios::binary);
    if (!lblFile.is_open()) {
        throw std::runtime_error("Unable to open label file");
    }

    // Read header for label file
    magicNumber = readInt32(lblFile);
    int numLabels = readInt32(lblFile);

    if (magicNumber != 2049) {
        throw std::runtime_error("Invalid MNIST label file format");
    }

    // Read label data
    data.labels.resize(numLabels);
    lblFile.read(reinterpret_cast<char*>(data.labels.data()), numLabels);

    return data;
}

/*
int main() {
    try {
        // Load the MNIST data (specify paths to the dataset files)
        MNISTData mnistData = loadMNIST(
            "MNIST/train-images.idx3-ubyte",
            "MNIST/train-labels.idx1-ubyte"
        );

        size_t l = 2;
        while (mnistData.labels[l] != 9) l++;

        std::cout << "Loaded " << mnistData.numImages << " images, each of size "
                  << mnistData.rows << "x" << mnistData.cols << std::endl;
        std::cout << "First label: " << static_cast<int>(mnistData.labels[l]) << std::endl;
        std::cout << "First image pixel values: ";


        // print ascii art version of the images
        printf("\r\n");
        for (int i = 0; i < 28; ++i) {  
            for (int j=0; j < 28; j++) {
                if (mnistData.images[l][i*28 + j] > 0) {
                    printf("X");
                } else {
                    printf(" ");
                }
            }
            printf("\r\n");
        }
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
*/
