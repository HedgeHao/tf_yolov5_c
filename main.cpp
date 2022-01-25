#include <iostream>
#include <vector>
#include <tensorflow/c/c_api.h>

std::vector<TF_Output> in_ops;
std::vector<TF_Output> out_ops;
TF_Session *sess;
std::vector<std::int64_t> input_dims;
std::vector<std::int64_t> output_dims;
int64_t input_size;
int64_t output_size;

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
    TF_Status *status = TF_NewStatus();
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

int main()
{
    printf("+++\n");
    loadModel();

    printf("---\n");
    return 0;
}