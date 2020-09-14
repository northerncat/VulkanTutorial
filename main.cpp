#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// helper function to print an array of string
void printCharPointerArray(const char** strings, int length, const std::string& label = "", std::ostream& output = std::cout) {
    output << label.c_str() << std::endl;
    for (int i = 0; i < length; ++i) {
        output << strings[i] << std::endl;
    }
    output << std::endl;
}

void printVkExtensions(const std::vector<VkExtensionProperties>& extensions) {
    std::cout << "Available Extensions: " << std::endl;
    for (const auto& extension : extensions) {
        std::cout << extension.extensionName << std::endl;
    }
    std::cout << std::endl;
}

void printVkLayers(const std::vector<VkLayerProperties>& layers) {
    std::cout << "Available Layers: " << std::endl;
    for (const auto& layer : layers) {
        std::cout << layer.layerName << std::endl;
    }
    std::cout << std::endl;

}

class VulkanTutorialApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
    }

    void createInstance() {

        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("Validation layers requested, but not available!");
        }

        // optional application info that could provide useful info for the driver to optimize
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // get desired global extensions from GLFW
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;
        printCharPointerArray(glfwExtensions, glfwExtensionCount);

        // populate the global validation layers to enable
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        // second parameter is a pointer to custom allocator callbacks
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("createInstance: Failed to create instance!");
        }

        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
        printVkExtensions(extensions);

    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        printVkLayers(availableLayers);

        return true;
    }

    GLFWwindow* window;

    VkInstance instance;
};

int main() {
    VulkanTutorialApplication app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
