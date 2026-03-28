struct MVP
{
    float4x4 model;
    float4x4 view;
    float4x4 projection;
};

[[vk::binding(0, 0)]]
ConstantBuffer<MVP> mvp;

struct VertexInput
{
    float2 position : POSITION;
    float3 color    : COLOR;
};

struct VertexOutput
{
    float4 sv_position : SV_Position;
    float3 color       : COLOR;
};

[shader("vertex")]
VertexOutput vertMain(VertexInput vin)
{
    VertexOutput output;
    float4 pos = float4(vin.position, 0.0, 1.0);
    output.sv_position = mul(mvp.projection, mul(mvp.view, mul(mvp.model, pos)));
    output.color       = vin.color;
    return output;
}

[shader("fragment")]
float4 fragMain(VertexOutput inVert)
    : SV_Target
{
    float3 color = inVert.color;
    return float4(color, 1.0);
}
