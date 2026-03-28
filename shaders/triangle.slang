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
    output.sv_position = float4(vin.position, 0.0, 1.0);
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
