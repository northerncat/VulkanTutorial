#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>
#include <set>

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
void printCharPointerArray(const char** strings, uint32_t length, const std::string& label = "", std::ostream& output = std::cout) {
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


VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger
) {
    // need to find the address of vkCreateDebugUtilsMessengerEXT since it's an extension function
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator
) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices {
    // queue family for drawing commands
    std::optional<uint32_t> graphicsFamily;
    // queue family for the device to present images to the created surface
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

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
        setupDebugMessenger();
        createSurface();

        pickPhysicalDevice();
        createLogicalDevice();
    }

    void createInstance() {

        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("createInstance: Validation layers requested, but not available!");
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

        // get desired global extensions from GLFW and debug callback
        auto requiredExtensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size());
        createInfo.ppEnabledExtensionNames = requiredExtensions.data();
        printCharPointerArray(requiredExtensions    .data(), requiredExtensions.size());

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo;
        // populate the global validation layers to enable
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            // pass in debugCreateInfo to the pNext field of instance creation to debug during
            // create/destroy of the instance
            populateDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }
        else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
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
        // note how logical devices are not directly related to the instance
        vkDestroyDevice(device, nullptr);

        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }

        vkDestroySurfaceKHR(instance, surface, nullptr);
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

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    // debug callback that takes the form of PFN_vkDebugUtilsMessengerCallbackEXT prototype to
    // report error if validation layer fails.
    //
    // messageSeverity: bit flag indicating the message severity, from verbose, info to warning and
    //     error. Could be compared to check if the severity is equal or more severe than another.
    // messageType: indicates whether the message is generic, related to specification/validation
    //     or related to potential performance issues from misuses.
    // pCallbackData: struct that contains detail of the message.
    // pUserData: passed in during callback setup to allow custom data.
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData
    ) {
        std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;

        // indicate whether the Vulkan call that triggers the debug callback should be aborted
        return VK_FALSE;
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }

    void setupDebugMessenger() {
        if (!enableValidationLayers) {
            return;
        }

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("setupDebugMessenger: Failed to set up debug messenger!");
        }
    }

    void createSurface() {
        // GLFW handles the platform specific logic for creating a VkSurfaceKHR; otherwise we
        // would create a Win32SurfaceKHR through VkWin32SurfaceCreateInfoKHR:
        //
        // VkWin32SurfaceCreateInfoKHR createInfo{};
        // createInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        // createInfo.hwnd = glfwGetWin32Window(window);
        // createInfo.hinstance = GetModuleHandle(nullptr);
        // vkCreateWin32SurfaceKHR(instance, &createInfo, nullptr, &surface) != VK_SUCCESS
        //
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("createSurface: Failed to create window surface!");
        }
    }

    // select graphics card in the system that supports desired features
    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("pickPhysicalDevice: Failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("pickPhysicalDevice: Failed to find a suitable GPU!");
        }
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        return indices.isComplete();
    }

    // check which queue families are supported by the device and which one of these supports the commands that we want to
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices;
        // assign index to queue families that could be found
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        uint32_t index = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VkQueueFlagBits::VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = index;
            }

            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &presentSupport);
            if (presentSupport) {
                indices.presentFamily = index;
            }

            if (indices.isComplete()) {
                break;
            }

            ++index;
        }

        return indices;
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            // specify which queue family the queue belongs to and the number of queues we want
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            // assign priority to queues for scheduling of command buffer execution, 0.0f - 1.0f
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // specify desired physical device features that was checked for in selecting a physical device
        VkPhysicalDeviceFeatures deviceFeatures{};

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.pEnabledFeatures = &deviceFeatures;

        // device creation ignores enabledExtensionCount and ppEnabledLayerNames because there was
        // distinction between instance and device specific layers; setting them is still good for
        // compatibility with older versions of Vulkan
        createInfo.enabledExtensionCount = 0;

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }

        // note how logical devices are not directly related to the instance
        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
            throw std::runtime_error("createLogicalDevice: Failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    GLFWwindow* window;

    VkInstance instance;

    VkDebugUtilsMessengerEXT debugMessenger;

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    // logical device is created to interface with the physical device
    VkDevice device;

    // queue handles to interface with the device's graphics and present queues
    VkQueue graphicsQueue;
    VkQueue presentQueue;

    // Windows System Integration (WSI) interface from the VK_KHR_surface extension
    VkSurfaceKHR surface;

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
