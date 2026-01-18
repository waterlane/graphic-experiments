#include <GL/glut.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

struct Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}

    Vec3 operator+(const Vec3 &o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
    Vec3 operator-(const Vec3 &o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }
};

static Vec3 operator*(float s, const Vec3 &v) { return Vec3(v.x * s, v.y * s, v.z * s); }

static float dot(const Vec3 &a, const Vec3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}
static Vec3 normalize(const Vec3 &v) {
    float len2 = dot(v, v);
    if (len2 <= 1e-8f) return Vec3(0, 0, 0);
    float inv = 1.0f / std::sqrt(len2);
    return v * inv;
}
static float length(const Vec3 &v) { return std::sqrt(dot(v, v)); }
static Vec3 clamp01(const Vec3 &v) {
    auto c = [](float x) { return x < 0 ? 0.f : (x > 1.f ? 1.f : x); };
    return Vec3(c(v.x), c(v.y), c(v.z));
}

struct Ray {
    Vec3 o; // origin
    Vec3 d; // direction (normalized)
};

struct Sphere {
    Vec3 center;
    float radius;
    Vec3 color;
};

// 房间是一个盒子：x ∈ [0, 5], y ∈ [0, 3], z ∈ [0, 5]
// 使用平面方程做相交测试
struct Plane {
    Vec3 n;   // 法线，指向房间内部
    float d;  // 平面方程 n·p + d = 0
    Vec3 color;
};

static int g_width = 800;
static int g_height = 600;

// 观察与光源参数（可交互修改）
static Vec3 g_camPos(2.5f, 1.5f, 8.0f);   // 摄像机位置
static Vec3 g_camLook(2.5f, 1.5f, 0.0f);  // 观察点
// 将光源放在相机一侧偏上方，让初始看到球的亮面
static Vec3 g_lightPos(2.5f, 3.0f, 6.0f); // 光源位置

// 场景数据
static std::vector<Sphere> g_spheres;
static std::vector<Plane> g_planes;
static std::vector<unsigned char> g_colorBuffer; // RGB buffer

static void init_scene() {
    // 两个球：红、蓝
    g_spheres.clear();
    const float sphereRadius = 0.9f;
    g_spheres.push_back(Sphere{Vec3(1.5f, sphereRadius, 2.5f), sphereRadius, Vec3(1.0f, 0.1f, 0.1f)}); // 红色
    g_spheres.push_back(Sphere{Vec3(3.5f, sphereRadius, 3.5f), sphereRadius, Vec3(0.1f, 0.1f, 1.0f)}); // 蓝色

    // 房间平面
    g_planes.clear();
    // 地面 y = 0，法线(0,1,0)，棕色
    g_planes.push_back(Plane{Vec3(0, 1, 0), 0.0f, Vec3(0.45f, 0.30f, 0.15f)});
    // 天花板 y = 3，法线(0,-1,0)
    g_planes.push_back(Plane{Vec3(0, -1, 0), 3.0f, Vec3(1.0f, 1.0f, 1.0f)});
    // 后墙 z = 0，法线(0,0,1)
    g_planes.push_back(Plane{Vec3(0, 0, 1), 0.0f, Vec3(1.0f, 1.0f, 1.0f)});
    // 右墙 x = 5，法线(-1,0,0)
    g_planes.push_back(Plane{Vec3(-1, 0, 0), 5.0f, Vec3(1.0f, 1.0f, 1.0f)});
    // 左墙 x = 0，法线(1,0,0)
    g_planes.push_back(Plane{Vec3(1, 0, 0), 0.0f, Vec3(1.0f, 1.0f, 1.0f)});

    g_colorBuffer.resize(g_width * g_height * 3);
}

static bool intersect_sphere(const Ray &ray, const Sphere &s, float &t, Vec3 &normal) {
    Vec3 oc = ray.o - s.center;
    float a = dot(ray.d, ray.d);
    float b = 2.0f * dot(oc, ray.d);
    float c = dot(oc, oc) - s.radius * s.radius;
    float disc = b * b - 4 * a * c;
    if (disc < 0.0f) return false;
    float sqrt_disc = std::sqrt(disc);
    float t0 = (-b - sqrt_disc) / (2 * a);
    float t1 = (-b + sqrt_disc) / (2 * a);
    float t_hit = t0;
    if (t_hit < 1e-4f) t_hit = t1;
    if (t_hit < 1e-4f) return false;
    t = t_hit;
    Vec3 hitPoint = ray.o + ray.d * t;
    normal = normalize(hitPoint - s.center);
    return true;
}

static bool intersect_plane(const Ray &ray, const Plane &pl, float &t, Vec3 &normal) {
    float denom = dot(pl.n, ray.d);
    if (std::fabs(denom) < 1e-6f) return false; // 平行
    float num = -(dot(pl.n, ray.o) + pl.d);
    float t_hit = num / denom;
    if (t_hit < 1e-4f) return false;

    Vec3 hitPoint = ray.o + ray.d * t_hit;
    // 房间限制：只保留盒子内部
    if (hitPoint.x < 0.0f - 1e-3f || hitPoint.x > 5.0f + 1e-3f ||
        hitPoint.y < 0.0f - 1e-3f || hitPoint.y > 3.0f + 1e-3f ||
        hitPoint.z < 0.0f - 1e-3f || hitPoint.z > 5.0f + 1e-3f)
        return false;

    t = t_hit;
    normal = pl.n; // 已经指向房间内部
    return true;
}

// 递归最大深度（用于反射）
static const int kMaxDepth = 2;

// 简单光照：Lambert 漫反射 + 阴影（硬阴影）+ 高光
static Vec3 shade(const Vec3 &hitPoint, const Vec3 &normal, const Vec3 &baseColor, const Vec3 &viewDir) {
    Vec3 L = normalize(g_lightPos - hitPoint);
    float lightDist = length(g_lightPos - hitPoint);

    // 阴影测试：从 hitPoint 沿 L 发一条光线，看是否被其它物体挡住
    Ray shadowRay;
    shadowRay.o = hitPoint + normal * 1e-3f; // 偏移一点点防止自相交
    shadowRay.d = L;

    bool inShadow = false;
    float tTmp;
    Vec3 nTmp;

    for (const auto &s : g_spheres) {
        if (intersect_sphere(shadowRay, s, tTmp, nTmp) && tTmp < lightDist - 1e-3f) {
            inShadow = true;
            break;
        }
    }
    if (!inShadow) {
        for (const auto &pl : g_planes) {
            if (intersect_plane(shadowRay, pl, tTmp, nTmp) && tTmp < lightDist - 1e-3f) {
                inShadow = true;
                break;
            }
        }
    }

    float ndotl = std::max(0.0f, dot(normal, L));
    float ambient = 0.2f;
    float diff = inShadow ? 0.0f : ndotl;

    // 漫反射
    Vec3 color = baseColor * (ambient + diff * 0.8f);

    // 高光（Blinn-Phong），让球看起来更“光滑”
    Vec3 H = normalize(L + viewDir);
    float ndoth = std::max(0.0f, dot(normal, H));
    float spec = inShadow ? 0.0f : std::pow(ndoth, 32.0f); // 降低高光锐度
    Vec3 specColor = Vec3(1.0f, 1.0f, 1.0f) * (spec * 0.3f); // 降低高光强度

    color = color + specColor;
    return clamp01(color);
}

static Vec3 trace(const Ray &ray, int depth);

static Vec3 trace(const Ray &ray) { return trace(ray, 0); }

static Vec3 trace(const Ray &ray, int depth) {
    float tMin = 1e30f;
    Vec3 hitNormal;
    Vec3 hitColor(0, 0, 0);
    float hitReflectivity = 0.0f; // 反射系数
    bool hit = false;

    // 先测球
    for (const auto &s : g_spheres) {
        float t;
        Vec3 n;
        if (intersect_sphere(ray, s, t, n) && t < tMin) {
            tMin = t;
            hitNormal = n;
            hitColor = s.color;
            hitReflectivity = 0.0f; // 只做局部光照，不做反射
            hit = true;
        }
    }

    // 再测平面
    for (const auto &pl : g_planes) {
        float t;
        Vec3 n;
        if (intersect_plane(ray, pl, t, n) && t < tMin) {
            tMin = t;
            hitNormal = n;
            hitColor = pl.color;
            hitReflectivity = 0.05f; // 这里设为不反射（需要的话可以改大一点）
            hit = true;
        }
    }

    if (!hit) {
        // 背景：稍微偏蓝的环境色
        return Vec3(0.2f, 0.3f, 0.5f);
    }

    Vec3 hitPoint = ray.o + ray.d * tMin;

    // 本地光照（漫反射 + 高光）
    Vec3 viewDir = normalize(ray.d * -1.0f);
    Vec3 localColor = shade(hitPoint, hitNormal, hitColor, viewDir);

    // 反射：让球在合适角度能“照到”墙面颜色
    if (depth < kMaxDepth && hitReflectivity > 0.0f) {
        Vec3 reflDir = normalize(ray.d - 2.0f * dot(ray.d, hitNormal) * hitNormal);
        Ray reflRay;
        reflRay.o = hitPoint + hitNormal * 1e-3f;
        reflRay.d = reflDir;

        Vec3 reflColor = trace(reflRay, depth + 1);
        localColor = (1.0f - hitReflectivity) * localColor + hitReflectivity * reflColor;
    }

    return clamp01(localColor);
}

static void render_scene() {
    // 简单固定相机：根据 g_camPos 和 g_camLook 构造视图平面
    Vec3 forward = normalize(g_camLook - g_camPos);
    Vec3 worldUp(0, 1, 0);
    Vec3 right = normalize(cross(forward, worldUp));
    // 处理 forward 与 worldUp 共线的极端情况
    if (dot(right, right) < 1e-6f) {
        worldUp = Vec3(0, 0, 1);
        right = normalize(cross(forward, worldUp));
    }
    Vec3 up = normalize(cross(right, forward));

    float fov = 45.0f * 3.1415926f / 180.0f;
    float aspect = static_cast<float>(g_width) / static_cast<float>(g_height);
    float scale = std::tan(fov * 0.5f);

    for (int y = 0; y < g_height; ++y) {
        for (int x = 0; x < g_width; ++x) {
            float u = (2.0f * ((x + 0.5f) / g_width) - 1.0f) * aspect * scale;
            float v = (2.0f * ((y + 0.5f) / g_height) - 1.0f) * scale;

            Vec3 dir = normalize(forward + u * right + v * up);
            Ray ray;
            ray.o = g_camPos;
            ray.d = dir;

            Vec3 col = trace(ray);
            col = clamp01(col);
            int idx = (y * g_width + x) * 3;
            g_colorBuffer[idx + 0] = static_cast<unsigned char>(col.x * 255.0f);
            g_colorBuffer[idx + 1] = static_cast<unsigned char>(col.y * 255.0f);
            g_colorBuffer[idx + 2] = static_cast<unsigned char>(col.z * 255.0f);
        }
    }
}

static void display_cb() {
    render_scene();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    glRasterPos2f(-1.f, -1.f);
    glDrawPixels(g_width, g_height, GL_RGB, GL_UNSIGNED_BYTE, g_colorBuffer.data());

    glutSwapBuffers();
}

static void reshape_cb(int w, int h) {
    if (w <= 0 || h <= 0) return;
    g_width = w;
    g_height = h;
    g_colorBuffer.resize(g_width * g_height * 3);
    glViewport(0, 0, w, h);
    glutPostRedisplay();
}

static void keyboard_cb(unsigned char key, int /*x*/, int /*y*/) {
    const float camStep = 0.3f;
    const float lightStep = 0.3f;

    switch (key) {
    case 27: // ESC
        std::exit(0);
        break;
    // 相机前后左右移动
    case 'w': g_camPos.z -= camStep; g_camLook.z -= camStep; break;
    case 's': g_camPos.z += camStep; g_camLook.z += camStep; break;
    case 'a': g_camPos.x -= camStep; g_camLook.x -= camStep; break;
    case 'd': g_camPos.x += camStep; g_camLook.x += camStep; break;
    case 'q': g_camPos.y += camStep; g_camLook.y += camStep; break;
    case 'e': g_camPos.y -= camStep; g_camLook.y -= camStep; break;
    // 光源移动
    case 'i': g_lightPos.z -= lightStep; break;
    case 'k': g_lightPos.z += lightStep; break;
    case 'j': g_lightPos.x -= lightStep; break;
    case 'l': g_lightPos.x += lightStep; break;
    case 'u': g_lightPos.y += lightStep; break;
    case 'o': g_lightPos.y -= lightStep; break;
    default:
        break;
    }

    std::cout << "Camera: (" << g_camPos.x << ", " << g_camPos.y << ", " << g_camPos.z
              << ")  Light: (" << g_lightPos.x << ", " << g_lightPos.y << ", " << g_lightPos.z << ")\n";

    glutPostRedisplay();
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(g_width, g_height);
    glutCreateWindow("Ray Tracing Room (WASDQE move , IJKLUO move light, ESC exit)");

    glClearColor(0.f, 0.f, 0.f, 1.f);

    init_scene();

    glutDisplayFunc(display_cb);
    glutReshapeFunc(reshape_cb);
    glutKeyboardFunc(keyboard_cb);

    std::cout << "===== Ray Tracing 控制说明 =====\n";
    std::cout << "W/S: 沿 z 轴前后移动相机\n";
    std::cout << "A/D: 沿 x 轴左右移动相机\n";
    std::cout << "Q/E: 沿 y 轴上下移动相机\n";
    std::cout << "I/K/J/L/U/O: 分别沿 z/x/y 轴移动光源\n";
    std::cout << "ESC: 退出程序\n";

    glutMainLoop();
    std::cout << "[Debug] glutMainLoop returned, program exiting.\n";
    return 0;
}
