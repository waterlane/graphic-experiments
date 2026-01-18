#pragma once
// Minimal OBJ loader for small samples (positions, normals, texcoords, triangles).
// Not a full replacement for the official tinyobjloader; meant for simple previews.
// Public domain / CC0-style.

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <iostream>

namespace tinyobj {

struct MeshData {
    std::vector<float> positions; // x,y,z per vertex
    std::vector<float> normals;   // nx,ny,nz per vertex (may be empty)
    std::vector<float> texcoords; // u,v per vertex (may be empty)
    std::vector<unsigned int> indices; // triangle indices
};

namespace detail {
inline bool starts_with(const std::string &s, const char *prefix) {
    const size_t len = std::char_traits<char>::length(prefix);
    return s.size() >= len && s.compare(0, len, prefix) == 0;
}

inline void trim(std::string &s) {
    const char *ws = " \t\r\n";
    auto b = s.find_first_not_of(ws);
    auto e = s.find_last_not_of(ws);
    if (b == std::string::npos) {
        s.clear();
        return;
    }
    s = s.substr(b, e - b + 1);
}

inline int to_index(int idx, size_t count) {
    if (idx > 0) return idx - 1;           // OBJ is 1-based
    if (idx < 0) return static_cast<int>(count) + idx; // negative indices
    return -1; // zero is invalid
}
}

inline bool LoadObj(MeshData &mesh, const std::string &filename, std::string &err) {
    std::ifstream ifs(filename);
    if (!ifs) {
        err = "Cannot open file: " + filename;
        return false;
    }

    std::vector<std::array<float, 3>> v_positions;
    std::vector<std::array<float, 3>> v_normals;
    std::vector<std::array<float, 2>> v_texcoords;

    std::unordered_map<std::string, unsigned int> vertex_map;

    auto add_vertex = [&](int vi, int ti, int ni) -> unsigned int {
        std::ostringstream key;
        key << vi << "/" << ti << "/" << ni;
        auto it = vertex_map.find(key.str());
        if (it != vertex_map.end()) return it->second;

        unsigned int new_index = static_cast<unsigned int>(mesh.positions.size() / 3);
        const auto &p = v_positions.at(static_cast<size_t>(vi));
        mesh.positions.insert(mesh.positions.end(), {p[0], p[1], p[2]});

        if (ni >= 0 && ni < static_cast<int>(v_normals.size())) {
            const auto &n = v_normals.at(static_cast<size_t>(ni));
            mesh.normals.insert(mesh.normals.end(), {n[0], n[1], n[2]});
        }

        if (ti >= 0 && ti < static_cast<int>(v_texcoords.size())) {
            const auto &t = v_texcoords.at(static_cast<size_t>(ti));
            mesh.texcoords.insert(mesh.texcoords.end(), {t[0], t[1]});
        }

        vertex_map.emplace(key.str(), new_index);
        return new_index;
    };

    std::string line;
    while (std::getline(ifs, line)) {
        detail::trim(line);
        if (line.empty() || line[0] == '#') continue;

        if (detail::starts_with(line, "v ")) {
            std::istringstream iss(line.substr(1));
            std::array<float, 3> v{};
            iss >> v[0] >> v[1] >> v[2];
            v_positions.push_back(v);
        } else if (detail::starts_with(line, "vn ")) {
            std::istringstream iss(line.substr(2));
            std::array<float, 3> n{};
            iss >> n[0] >> n[1] >> n[2];
            v_normals.push_back(n);
        } else if (detail::starts_with(line, "vt ")) {
            std::istringstream iss(line.substr(2));
            std::array<float, 2> t{};
            iss >> t[0] >> t[1];
            v_texcoords.push_back(t);
        } else if (detail::starts_with(line, "f ")) {
            std::istringstream iss(line.substr(1));
            std::vector<std::string> tokens;
            std::string tok;
            while (iss >> tok) tokens.push_back(tok);
            if (tokens.size() < 3) continue;

            auto parse_vertex = [&](const std::string &tok) -> std::tuple<int, int, int> {
                int vi = -1, ti = -1, ni = -1;
                std::stringstream ss(tok);
                std::string item;
                int field = 0;
                while (std::getline(ss, item, '/')) {
                    if (!item.empty()) {
                        int value = std::stoi(item);
                        if (field == 0) vi = detail::to_index(value, v_positions.size());
                        else if (field == 1) ti = detail::to_index(value, v_texcoords.size());
                        else if (field == 2) ni = detail::to_index(value, v_normals.size());
                    }
                    ++field;
                }
                return {vi, ti, ni};
            };

            std::vector<unsigned int> face_indices;
            face_indices.reserve(tokens.size());
            for (const auto &t : tokens) {
                auto [vi, ti, ni] = parse_vertex(t);
                if (vi < 0 || vi >= static_cast<int>(v_positions.size())) {
                    err = "Invalid vertex index in face";
                    return false;
                }
                unsigned int idx = add_vertex(vi, ti, ni);
                face_indices.push_back(idx);
            }

            // Triangulate (fan)
            for (size_t i = 1; i + 1 < face_indices.size(); ++i) {
                mesh.indices.push_back(face_indices[0]);
                mesh.indices.push_back(face_indices[i]);
                mesh.indices.push_back(face_indices[i + 1]);
            }
        }
    }

    return true;
}

} // namespace tinyobj
