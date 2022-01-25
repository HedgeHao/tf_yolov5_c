#include <iostream>
#include <vector>
#include <tensorflow/c/c_api.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>

TF_Status *status;
std::vector<TF_Output> in_ops;
std::vector<TF_Output> out_ops;
TF_Session *sess;
std::vector<std::int64_t> input_dims;
std::vector<std::int64_t> output_dims;
int64_t input_size;
int64_t output_size;
std::vector<TF_Tensor *> in_tensors;
std::vector<TF_Tensor *> out_tensors;

static void DeallocateBuffer(void *data, size_t)
{
    std::free(data);
}

static TF_Buffer *ReadBufferFromFile(const char *file)
{
    const auto f = std::fopen(file, "rb");
    if (f == nullptr)
    {
        return nullptr;
    }

    std::fseek(f, 0, SEEK_END);
    const auto fsize = ftell(f);
    std::fseek(f, 0, SEEK_SET);

    if (fsize < 1)
    {
        std::fclose(f);
        return nullptr;
    }

    const auto data = std::malloc(fsize);
    std::fread(data, fsize, 1, f);
    std::fclose(f);

    TF_Buffer *buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = DeallocateBuffer;

    return buf;
}

int loadModel()
{
    TF_Buffer *buffer = ReadBufferFromFile("yolov5n.pb");

    TF_Graph *graph = TF_NewGraph();
    status = TF_NewStatus();
    TF_ImportGraphDefOptions *opts = TF_NewImportGraphDefOptions();

    TF_GraphImportGraphDef(graph, buffer, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(buffer);

    if (TF_GetCode(status) != TF_OK)
    {
        TF_DeleteGraph(graph);
        graph = nullptr;
    }
    TF_DeleteStatus(status);

    if (graph == nullptr)
    {
        std::cout << "Can't load graph" << std::endl;
        return 1;
    }
    status = TF_NewStatus();

    // create session
    TF_SessionOptions *options = TF_NewSessionOptions();
    sess = TF_NewSession(graph, options, status);
    TF_DeleteSessionOptions(options);
    if (TF_GetCode(status) != TF_OK)
    {
        TF_DeleteStatus(status);
        return 2;
    }

    // I/O operatiors
    in_ops.push_back({TF_GraphOperationByName(graph, "x"), 0});
    out_ops.push_back({TF_GraphOperationByName(graph, "Identity"), 0});
    for (size_t i = 0; i < in_ops.size(); i++)
    {
        if (in_ops[i].oper == nullptr)
        {
            std::cout << "Can't init input_op" << std::endl;
        }
    }
    for (size_t i = 0; i < out_ops.size(); i++)
    {
        if (out_ops[i].oper == nullptr)
        {
            std::cout << "Can't init output_op" << std::endl;
        }
    }

    input_dims = {1, 640, 640, 3};
    output_dims = {1, 25200, 85};
    input_size = 1;
    for (int i = 0; i < input_dims.size(); i++)
    {
        input_size *= input_dims[i];
    }

    output_size = 1;
    for (int i = 0; i < output_dims.size(); i++)
    {
        output_size *= output_dims[i];
    }
}

TF_Tensor *CreateTensor(TF_DataType data_type,
                        const std::int64_t *dims, std::size_t num_dims,
                        const void *data, std::size_t len)
{
    if (dims == nullptr || data == nullptr)
        return nullptr;

    TF_Tensor *tensor = TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), len);
    if (tensor == nullptr)
        return nullptr;

    void *tensor_data = TF_TensorData(tensor);
    if (tensor_data == nullptr)
    {
        TF_DeleteTensor(tensor);
        return nullptr;
    }

    if (len > TF_TensorByteSize(tensor))
    {
        std::memcpy(tensor_data, data, len);
    }
    else
    {
        std::memcpy(tensor_data, data, TF_TensorByteSize(tensor));
    }

    return tensor;
}

int executeModel(std::vector<TF_Output> input_ops, std::vector<TF_Output> output_ops,
                 TF_Tensor **input_tensors, TF_Tensor **output_tensors)
{
    TF_SessionRun(sess,                                                 // Session
                  nullptr,                                              // Run options.
                  input_ops.data(), input_tensors, input_ops.size(),    // Input tensors, input tensor values, number of inputs.
                  output_ops.data(), output_tensors, output_ops.size(), // Output tensors, output tensor values, number of outputs.
                  nullptr, 0,                                           // Target operations, number of targets.
                  nullptr,                                              // Run metadata.
                  status                                                // Output status.
    );
    if (TF_GetCode(status) != TF_OK)
    {
        std::cout << "Error run session";
        TF_DeleteStatus(status);
        return 5;
    }

    return 0;
}

void DeleteTensor(TF_Tensor *tensor)
{
    if (tensor == nullptr)
    {
        return;
    }
    TF_DeleteTensor(tensor);
}

std::vector<float> pipeline(cv::Mat img)
{
    std::vector<float> data(input_size);
    float *ptr_data = data.data();

    cv::resize(img, img, cv::Size(int(input_dims[3]), int(input_dims[2])), 0, 0, cv::INTER_LINEAR);
    for (int i = 0; i < input_dims[2]; i++)
    {
        uint8_t *ptr_img = img.ptr<uint8_t>(i);
        for (int j = 0; j < input_dims[3]; j++)
        {
            *ptr_data++ = static_cast<float>(*ptr_img++);
        }
    }

    TF_Tensor *input_tensor = CreateTensor(TF_FLOAT, input_dims.data(), input_dims.size(), data.data(), data.size() * sizeof(float));
    TF_Tensor *pred_tensor = nullptr;
    in_tensors.push_back(input_tensor);
    out_tensors.push_back(pred_tensor);

    executeModel(in_ops, out_ops, in_tensors.data(), out_tensors.data());

    float *out = static_cast<float *>(TF_TensorData(out_tensors[0]));
    std::vector<float> feature(out, out + output_size);

    for (int i = 0; i < 100; i++)
    {
        printf("%.2f, ", feature[i]);
    }
    printf("\n");

    // 4. delete input & output tensors
    while (!in_tensors.empty())
    {
        DeleteTensor(in_tensors.back());
        in_tensors.pop_back();
    }
    while (!out_tensors.empty())
    {
        DeleteTensor(out_tensors.back());
        out_tensors.pop_back();
    }

    return feature;
}

int main()
{
    printf("+++\n");
    loadModel();

    char *file = "../images/bus.jpg";
    cv::Mat img = cv::imread(file, 1);

    // cv::namedWindow("Preview", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Preview", img);
    // cv::waitKey(0);

    pipeline(img);

    printf("---\n");
    return 0;
}