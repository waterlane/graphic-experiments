#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "../third_party/tiny_obj_loader.h"

struct Mat4 {
    float m[16]; // column-major
};

static Mat4 identity() {
    Mat4 r{};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

static Mat4 perspective(float fovy, float aspect, float znear, float zfar) {
    float f = 1.0f / std::tan(fovy * 0.5f);
    Mat4 r{};
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zfar + znear) / (znear - zfar);
    r.m[11] = -1.0f;
    r.m[14] = (2.0f * zfar * znear) / (znear - zfar);
    return r;
}

static Mat4 translate(float x, float y, float z) {
    Mat4 r = identity();
    r.m[12] = x;
    r.m[13] = y;
    r.m[14] = z;
    return r;
}

static Mat4 rotate_y(float angle_rad) {
    Mat4 r = identity();
    float c = std::cos(angle_rad);
    float s = std::sin(angle_rad);
    r.m[0] = c;
    r.m[2] = s;
    r.m[8] = -s;
    r.m[10] = c;
    return r;
}

static Mat4 rotate_x(float angle_rad) {
    Mat4 r = identity();
    float c = std::cos(angle_rad);
    float s = std::sin(angle_rad);
    r.m[5] = c;
    r.m[6] = s;
    r.m[9] = -s;
    r.m[10] = c;
    return r;
}

static Mat4 scale(float s) {
    Mat4 r{};
    r.m[0] = r.m[5] = r.m[10] = s;
    r.m[15] = 1.0f;
    return r;
}

static Mat4 multiply(const Mat4 &a, const Mat4 &b) {
    Mat4 r{};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            r.m[col * 4 + row] =
                a.m[0 * 4 + row] * b.m[col * 4 + 0] +
                a.m[1 * 4 + row] * b.m[col * 4 + 1] +
                a.m[2 * 4 + row] * b.m[col * 4 + 2] +
                a.m[3 * 4 + row] * b.m[col * 4 + 3];
        }
    }
    return r;
}

struct InteractionState {
    bool rotating = false;
    bool panning = false;
    double last_x = 0.0;
    double last_y = 0.0;
    float yaw = 0.0f;
    float pitch = 0.0f;
    float pan_x = 0.0f;
    float pan_y = 0.0f;
    float distance = 3.0f;
    float model_scale = 1.0f;
};

static GLuint compile_shader(GLenum type, const char *src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0');
        glGetShaderInfoLog(shader, len, nullptr, log.data());
        std::cerr << "Shader compile error: " << log << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

static GLuint create_program(const char *vs_src, const char *fs_src) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    if (!vs || !fs) return 0;
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0');
        glGetProgramInfoLog(prog, len, nullptr, log.data());
        std::cerr << "Program link error: " << log << std::endl;
        glDeleteProgram(prog);
        prog = 0;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

static void compute_normals_if_missing(tinyobj::MeshData &mesh) {
    if (!mesh.normals.empty()) return;
    mesh.normals.assign(mesh.positions.size(), 0.0f);
    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        unsigned int ia = mesh.indices[i];
        unsigned int ib = mesh.indices[i + 1];
        unsigned int ic = mesh.indices[i + 2];
        const float *a = &mesh.positions[ia * 3];
        const float *b = &mesh.positions[ib * 3];
        const float *c = &mesh.positions[ic * 3];
        float ux = b[0] - a[0];
        float uy = b[1] - a[1];
        float uz = b[2] - a[2];
        float vx = c[0] - a[0];
        float vy = c[1] - a[1];
        float vz = c[2] - a[2];
        float nx = uy * vz - uz * vy;
        float ny = uz * vx - ux * vz;
        float nz = ux * vy - uy * vx;
        mesh.normals[ia * 3 + 0] += nx;
        mesh.normals[ia * 3 + 1] += ny;
        mesh.normals[ia * 3 + 2] += nz;
        mesh.normals[ib * 3 + 0] += nx;
        mesh.normals[ib * 3 + 1] += ny;
        mesh.normals[ib * 3 + 2] += nz;
        mesh.normals[ic * 3 + 0] += nx;
        mesh.normals[ic * 3 + 1] += ny;
        mesh.normals[ic * 3 + 2] += nz;
    }
    for (size_t i = 0; i + 2 < mesh.normals.size(); i += 3) {
        float nx = mesh.normals[i];
        float ny = mesh.normals[i + 1];
        float nz = mesh.normals[i + 2];
        float len = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (len > 1e-6f) {
            mesh.normals[i] /= len;
            mesh.normals[i + 1] /= len;
            mesh.normals[i + 2] /= len;
        }
    }
}

static void glfw_error_callback(int code, const char *desc) {
    std::cerr << "GLFW error " << code << ": " << desc << std::endl;
}

static void mouse_button_callback(GLFWwindow *window, int button, int action, int /*mods*/) {
    auto *state = static_cast<InteractionState *>(glfwGetWindowUserPointer(window));
    if (!state) return;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        state->rotating = (action == GLFW_PRESS);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        state->panning = (action == GLFW_PRESS);
    }
    if (action == GLFW_PRESS) {
        glfwGetCursorPos(window, &state->last_x, &state->last_y);
    }
}

static void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos) {
    auto *state = static_cast<InteractionState *>(glfwGetWindowUserPointer(window));
    if (!state) return;
    double dx = xpos - state->last_x;
    double dy = ypos - state->last_y;
    state->last_x = xpos;
    state->last_y = ypos;

    if (state->rotating) {
        state->yaw += static_cast<float>(dx * 0.005);
        state->pitch += static_cast<float>(dy * 0.005);
        state->pitch = std::clamp(state->pitch, -1.5f, 1.5f);
    }
    if (state->panning) {
        float factor = state->distance * 0.0025f;
        state->pan_x += static_cast<float>(dx) * factor;
        state->pan_y -= static_cast<float>(dy) * factor;
    }
}

static void scroll_callback(GLFWwindow *window, double /*xoffset*/, double yoffset) {
    auto *state = static_cast<InteractionState *>(glfwGetWindowUserPointer(window));
    if (!state) return;
    float scale = std::exp(static_cast<float>(-yoffset) * 0.1f);
    state->distance = std::clamp(state->distance * scale, 0.5f, 50.0f);
}

static std::pair<Mat4, Mat4> compute_matrices(const InteractionState &st, float aspect) {
    Mat4 proj = perspective(45.0f * 3.1415926f / 180.0f, aspect, 0.05f, 200.0f);
    Mat4 view = translate(0.0f, 0.0f, -st.distance);
    Mat4 t = translate(st.pan_x, st.pan_y, 0.0f);
    Mat4 ry = rotate_y(st.yaw);
    Mat4 rx = rotate_x(st.pitch);
    Mat4 s = scale(st.model_scale);

    Mat4 model = multiply(t, multiply(ry, multiply(rx, s)));
    Mat4 vp = multiply(proj, view);
    Mat4 mvp = multiply(vp, model);
    return {mvp, model};
}

int main(int argc, char **argv) {
    std::string obj_path = (argc > 1) ? argv[1] : "assets/cube.obj";

    tinyobj::MeshData mesh;
    std::string err;
    if (!tinyobj::LoadObj(mesh, obj_path, err)) {
        std::cerr << "Failed to load OBJ: " << err << std::endl;
        return 1;
    }
    if (mesh.indices.empty()) {
        std::cerr << "OBJ has no faces: " << obj_path << std::endl;
        return 1;
    }

    compute_normals_if_missing(mesh);

    // Interleave position and normal (6 floats per vertex)
    std::vector<float> interleaved;
    interleaved.reserve(mesh.positions.size() * 2);
    size_t vertex_count = mesh.positions.size() / 3;
    for (size_t i = 0; i < vertex_count; ++i) {
        interleaved.push_back(mesh.positions[i * 3 + 0]);
        interleaved.push_back(mesh.positions[i * 3 + 1]);
        interleaved.push_back(mesh.positions[i * 3 + 2]);
        interleaved.push_back(mesh.normals[i * 3 + 0]);
        interleaved.push_back(mesh.normals[i * 3 + 1]);
        interleaved.push_back(mesh.normals[i * 3 + 2]);
    }

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(1280, 720, "OBJ Preview", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    InteractionState state;
    glfwSetWindowUserPointer(window, &state);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return 1;
    }

    glEnable(GL_DEPTH_TEST);

    const char *vs_src = R"( #version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 vNormal;

void main() {
    vNormal = mat3(u_model) * aNormal;
    gl_Position = u_mvp * vec4(aPos, 1.0);
}
)";

    const char *fs_src = R"( #version 330 core
in vec3 vNormal;
out vec4 FragColor;

// 纯绿色物体
uniform vec3 u_color = vec3(0.0, 1.0, 0.0);

void main() {
    // 归一化法线和光方向（光源固定在世界/摄像机空间）
    vec3 N = normalize(vNormal);
    vec3 L = normalize(vec3(0.3, 1.0, 0.2));

    // 单侧漫反射，法线转到背面时亮度减弱
    float ndl = max(dot(N, L), 0.0);

    // 保证没有完全黑的面：亮度范围 [0.4, 1.0]
    float brightness = 0.4 + 0.6 * ndl;

    vec3 color = u_color * brightness;
    FragColor = vec4(color, 1.0);
}
)";

    GLuint program = create_program(vs_src, fs_src);
    if (!program) return 1;

    GLuint vao = 0, vbo = 0, ebo = 0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, interleaved.size() * sizeof(float), interleaved.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.size() * sizeof(unsigned int), mesh.indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        // Keyboard controls for fine adjustments
        float pan_step = 0.015f * state.distance;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) state.pan_x -= pan_step;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) state.pan_x += pan_step;
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) state.pan_y += pan_step;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) state.pan_y -= pan_step;
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) state.distance = std::clamp(state.distance * 0.99f, 0.5f, 50.0f);
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) state.distance = std::clamp(state.distance * 1.01f, 0.5f, 50.0f);

        if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS)
            state.model_scale = std::clamp(state.model_scale * 1.01f, 0.05f, 20.0f);
        if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS)
            state.model_scale = std::clamp(state.model_scale * 0.99f, 0.05f, 20.0f);

        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            state = InteractionState{};
        }

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        float aspect = (height == 0) ? 1.0f : static_cast<float>(width) / static_cast<float>(height);
        auto [mvp, model] = compute_matrices(state, aspect);

        glViewport(0, 0, width, height);
        glClearColor(0.07f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(program);
        GLint loc_mvp = glGetUniformLocation(program, "u_mvp");
        GLint loc_model = glGetUniformLocation(program, "u_model");
        glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp.m);
        glUniformMatrix4fv(loc_model, 1, GL_FALSE, model.m);

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(mesh.indices.size()), GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

    glDeleteProgram(program);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
