#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <3dmath.h>


struct Mesh
{
  const uint32_t vertexArrayBufferObject;
  const int numIndices;

  struct Bone
  {
    std::string name;
    std::optional<std::string> parentName;
    glm::mat4x4 bindPose, invBindPose;
  };

  std::vector<Bone> bones;
  std::unordered_map<std::string, int> bonesMap;

  Mesh(uint32_t vertexArrayBufferObject, int numIndices) :
    vertexArrayBufferObject(vertexArrayBufferObject),
    numIndices(numIndices)
    {}
};

using MeshPtr = std::shared_ptr<Mesh>;

MeshPtr load_mesh(const char *path, int idx);
MeshPtr make_plane_mesh();
MeshPtr make_mesh(const std::vector<uint32_t> &indices, const std::vector<vec3> &vertices, const std::vector<vec3> &normals);

void render(const MeshPtr &mesh);
void render(const MeshPtr &mesh, int count);