struct MVP
{
    float4x4 model;
    float4x4 view;
    float4x4 projection;
};

struct Vertex
{
    float3 position;
    float3 normal;
    float2 texCoord;
};

struct PushData
{
    Vertex *pVertexBuffer;
    MVP *pMvp;
};

[[vk::push_constant]]
PushData pushData;

[[vk::binding(0, 0)]]
Sampler2D texture;

struct VertexOutput
{
    float4 sv_position : SV_Position;
    float3 normal      : NORMAL;
    float2 texCoord    : TEXCOORD;
};

[shader("vertex")]
VertexOutput vertMain(uint vertexID : SV_VertexID)
{
    Vertex vin = pushData.pVertexBuffer[vertexID];
    MVP matrices = *pushData.pMvp;

    VertexOutput output;
    float4 pos = float4(vin.position, 1.0);
    output.sv_position = mul(matrices.projection, mul(matrices.view, mul(matrices.model, pos)));
    output.normal      = vin.normal;
    output.texCoord    = vin.texCoord;
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput inVert)
    : SV_Target
{
    float3 texColor = texture.Sample(inVert.texCoord).rgb;
    float3 lightDir = normalize(float3(1.0, 1.0, 0.5));
    float ndotl = max(dot(normalize(inVert.normal), lightDir), 0.15);
    return float4(texColor * ndotl, 1.0);
}
