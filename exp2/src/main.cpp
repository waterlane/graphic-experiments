#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "../third_party/tiny_obj_loader.h"

struct Vec3 {
    float x, y, z;
};

struct Quat {
    float w, x, y, z;
};

struct Mat4 {
    float m[16];
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

static Mat4 translate(const Vec3 &t) {
    Mat4 r = identity();
    r.m[12] = t.x;
    r.m[13] = t.y;
    r.m[14] = t.z;
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

// 四元数工具
static Quat quat_normalize(const Quat &q) {
    float len = std::sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (len < 1e-6f) return Quat{1,0,0,0};
    float inv = 1.0f / len;
    return Quat{q.w*inv, q.x*inv, q.y*inv, q.z*inv};
}

static Quat quat_from_axis_angle(const Vec3 &axis, float angle) {
    float half = angle * 0.5f;
    float s = std::sin(half);
    Vec3 na = axis;
    float len = std::sqrt(na.x*na.x + na.y*na.y + na.z*na.z);
    if (len < 1e-6f) return Quat{1,0,0,0};
    na.x /= len; na.y /= len; na.z /= len;
    return quat_normalize(Quat{std::cos(half), na.x*s, na.y*s, na.z*s});
}

static float quat_dot(const Quat &a, const Quat &b) {
    return a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
}

static Quat quat_slerp(Quat a, Quat b, float t) {
    a = quat_normalize(a);
    b = quat_normalize(b);
    float cos_om = quat_dot(a, b);
    if (cos_om < 0.0f) { b.w = -b.w; b.x = -b.x; b.y = -b.y; b.z = -b.z; cos_om = -cos_om; }
    const float EPS = 1e-5f;
    float k0, k1;
    if (1.0f - cos_om < EPS) {
        k0 = 1.0f - t;
        k1 = t;
    } else {
        float om = std::acos(cos_om);
        float inv_sin = 1.0f / std::sin(om);
        k0 = std::sin((1.0f - t) * om) * inv_sin;
        k1 = std::sin(t * om) * inv_sin;
    }
    return Quat{
        k0*a.w + k1*b.w,
        k0*a.x + k1*b.x,
        k0*a.y + k1*b.y,
        k0*a.z + k1*b.z
    };
}

static Mat4 quat_to_mat4(const Quat &q_in) {
    Quat q = quat_normalize(q_in);
    float w = q.w, x = q.x, y = q.y, z = q.z;
    Mat4 r = identity();
    r.m[0] = 1 - 2*y*y - 2*z*z;
    r.m[1] = 2*x*y + 2*w*z;
    r.m[2] = 2*x*z - 2*w*y;

    r.m[4] = 2*x*y - 2*w*z;
    r.m[5] = 1 - 2*x*x - 2*z*z;
    r.m[6] = 2*y*z + 2*w*x;

    r.m[8] = 2*x*z + 2*w*y;
    r.m[9] = 2*y*z - 2*w*x;
    r.m[10] = 1 - 2*x*x - 2*y*y;
    return r;
}

struct InteractionState {
    bool playing = false;      // 是否正在播放
    bool loop = false;         // 是否循环播放
    bool request_start = false;// 是否请求从头播放一次
    float time = 0.0f;     // [0,1] 插值参数
    float duration = 5.0f; // 动画周期（秒）
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
    if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS) return;
    auto *state = static_cast<InteractionState *>(glfwGetWindowUserPointer(window));
    if (!state) return;
    state->request_start = true; // 单击左键：请求播放一次动画
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

    GLFWwindow *window = glfwCreateWindow(1280, 720, "Quaternion Path Demo - press left mouse or space button to play animation once", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window" << std::endl;
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

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

uniform vec3 u_color;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(vec3(0.3, 1.0, 0.2));
    float ndl = max(dot(N, L), 0.0);
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

    // 定义起始和终止位姿
    Vec3 pos_start{-1.5f, 0.0f, 0.0f};
    Vec3 pos_end{1.5f, 0.5f, 0.0f};

    Quat ori_start = quat_from_axis_angle(Vec3{0,1,0}, 0.0f);           // 初始朝向
    Quat ori_end   = quat_from_axis_angle(Vec3{0,1,0}, 3.1415926f);     // 终止：绕 y 轴 180 度

    InteractionState state; // 默认不播放，等待用户触发
    glfwSetWindowUserPointer(window, &state);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    std::cout << "===== 交互说明 =====" << std::endl;
    std::cout << "左键单击窗口：从起始姿态到终止姿态播放一次平移+旋转动画" << std::endl;
    std::cout << "空格键：同上，从头播放一次动画" << std::endl;
    std::cout << "L 键：开启循环播放" << std::endl;
    std::cout << "K 键：关闭循环播放" << std::endl;
    std::cout << "R 键：重置到起始状态并停止播放" << std::endl;
    std::cout << "Esc：退出程序" << std::endl;
    double last_time = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }

        // 按键交互：
        // 空格：从头播放一次动画
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            state.time = 0.0f;
            state.playing = true;
        }
        // L：开启循环播放；K：关闭循环
        if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) state.loop = true;
        if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) state.loop = false;
        // R：重置并停止
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            state.time = 0.0f;
            state.playing = false;
        }

        // 鼠标左键点击请求：从头播放一次
        if (state.request_start) {
            state.time = 0.0f;
            state.playing = true;
            state.request_start = false;
        }

        double now = glfwGetTime();
        float dt = static_cast<float>(now - last_time);
        last_time = now;

        if (state.playing) {
            state.time += dt / state.duration;
            if (state.time >= 1.0f) {
                if (state.loop) {
                    // 循环播放
                    state.time -= 1.0f;
                } else {
                    // 一次性播放：到终点后停止
                    state.time = 1.0f;
                    state.playing = false;
                }
            }
        }

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        float aspect = (height == 0) ? 1.0f : static_cast<float>(width) / static_cast<float>(height);

        Mat4 proj = perspective(45.0f * 3.1415926f / 180.0f, aspect, 0.05f, 50.0f);
        Mat4 view = identity();
        view.m[14] = -6.0f; // 简单后移摄像机

        glViewport(0, 0, width, height);
        glClearColor(0.07f, 0.08f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(program);
        GLint loc_mvp = glGetUniformLocation(program, "u_mvp");
        GLint loc_model = glGetUniformLocation(program, "u_model");
        GLint loc_color = glGetUniformLocation(program, "u_color");

        glBindVertexArray(vao);

        auto draw_pose = [&](const Vec3 &pos, const Quat &ori, float s, float r, float g, float b) {
            Mat4 t = translate(pos);
            Mat4 rmat = quat_to_mat4(ori);
            Mat4 smat = scale(s);
            Mat4 model = multiply(t, multiply(rmat, smat));
            Mat4 vp = multiply(proj, view);
            Mat4 mvp = multiply(vp, model);
            glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp.m);
            glUniformMatrix4fv(loc_model, 1, GL_FALSE, model.m);
            glUniform3f(loc_color, r,g,b);
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(mesh.indices.size()), GL_UNSIGNED_INT, nullptr);
        };

        // 1) 起始姿态：蓝色
        draw_pose(pos_start, ori_start, 1.0f, 0.0f, 0.6f, 1.0f);

        // 1) 终止姿态：红色
        draw_pose(pos_end, ori_end, 1.0f, 1.0f, 0.2f, 0.2f);

        // 2/3) 平滑平移 + 旋转：绿色（中间插值）
        float t_interp = std::clamp(state.time, 0.0f, 1.0f);
        Vec3 pos_mid{
            pos_start.x + (pos_end.x - pos_start.x) * t_interp,
            pos_start.y + (pos_end.y - pos_start.y) * t_interp,
            pos_start.z + (pos_end.z - pos_start.z) * t_interp
        };
        Quat ori_mid = quat_slerp(ori_start, ori_end, t_interp);

        draw_pose(pos_mid, ori_mid, 1.0f, 0.0f, 1.0f, 0.0f);

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
