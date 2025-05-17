#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D screenTex;
void main() {
    vec2 uv = vec2 (TexCoord.x, 1.0 - TexCoord.y); // Flip Y coordinate
    FragColor = texture(screenTex, uv);
}
