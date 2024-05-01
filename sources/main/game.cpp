#include <render/direction_light.h>
#include <render/material.h>
#include <render/mesh.h>
#include <render/scene.h>
#include "camera.h"
#include <my_time.h>
#include <application.h>
#include <render/debug_arrow.h>
#include <render/debug_bone.h>
#include <imgui/imgui.h>
#include "ImGuizmo.h"
#include "ComboWithFilter.h"

#include "ozz/animation/runtime/animation.h"
#include "ozz/animation/runtime/local_to_model_job.h"
#include "ozz/animation/runtime/sampling_job.h"
#include "ozz/animation/runtime/skeleton.h"
#include "ozz/base/maths/simd_math.h"
#include "ozz/base/maths/soa_transform.h"
#include "ozz/animation/offline/animation_optimizer.h"
#include "ozz/animation/runtime/blending_job.h"

struct UserCamera
{
  glm::mat4 transform;
  mat4x4 projection;
  vec2 windowSize;
  ArcballCamera arcballCamera;
};

struct PlaybackController
{
public:
  // Updates animation time if in "play" state, according to playback speed and
  // given frame time _dt.
  // Returns true if animation has looped during update
  void Update(const AnimationPtr &_animation, float _dt)
  {
    float new_time = time_ratio_;

    if (play_)
    {
      new_time = time_ratio_ + _dt * playback_speed_ / _animation->duration();
    }
    if (loop_)
    {
      // Wraps in the unit interval [0:1], even for negative values (the reason
      // for using floorf).
      time_ratio_ = new_time - floorf(new_time);
    }
    else
    {
      // Clamps in the unit interval [0:1].
      time_ratio_ = new_time;
    }
  }

  void Reset()
  {
    time_ratio_ = 0.f;
    playback_speed_ = 1.f;
    play_ = true;
    loop_ = true;
  }

  // Current animation time ratio, in the unit interval [0,1], where 0 is the
  // beginning of the animation, 1 is the end.
  float time_ratio_;

  // Playback speed, can be negative in order to play the animation backward.
  float playback_speed_;

  // Animation play mode state: play/pause.
  bool play_;

  // Animation loop mode.
  bool loop_;
};

struct OptimizerController {
  float tolerance = 0;
  float distance = 0;
  bool optimized = false;
  AnimationInfo animation_info;
};

struct AnimationLayer
{
  // Constructor, default initialization.
  AnimationLayer(const SkeletonPtr &skeleton, AnimationPtr animation, int animationIndex, const AnimationInfo &animation_info) : 
          weight(1.f), animation(animation), animationIndex(animationIndex)
  {
    controller.Reset();
    optimizer.animation_info = animation_info;
    locals.resize(skeleton->num_soa_joints());

    // Allocates a context that matches animation requirements.
    context = std::make_shared<ozz::animation::SamplingJob::Context>(skeleton->num_joints());
  }

  bool isAdditive = false;
  // Playback animation controller. This is a utility class that helps with
  // controlling animation playback time.
  PlaybackController controller;
  OptimizerController optimizer; 
  int animationIndex = -1;
  // Blending weight for the layer.
  float weight;

  // Runtime animation.
  AnimationPtr animation;

  // Sampling context.
  std::shared_ptr<ozz::animation::SamplingJob::Context> context;

  // Buffer of local transforms as sampled from animation_.
  std::vector<ozz::math::SoaTransform> locals;
  bool visability = true;
  bool owned_by_blendtree = false;
};

struct BlendTree1d {
  float weight = -1;
  int layersId[3] = {-1, -1, -1};
  size_t nLayers = 0;
};

struct Character
{
  glm::mat4 transform;
  MeshPtr mesh;
  MaterialPtr material;

  SkeletonPtr skeleton_;
  std::shared_ptr<ozz::animation::SamplingJob::Context> context_;

  // Buffer of local transforms as sampled from animation_.
  std::vector<ozz::math::SoaTransform> locals_;

  // Buffer of model space matrices.
  std::vector<ozz::math::Float4x4> models_;

  std::vector<AnimationLayer> layers;

  AnimationPtr currentAnimation;
  PlaybackController controller;
  OptimizerController optimizer;
  int animationIndex = -1;

  std::vector<BlendTree1d> blendtrees;
};

struct Scene
{
  DirectionLight light;

  UserCamera userCamera;

  std::vector<Character> characters;

};

static glm::mat4 to_glm(const ozz::math::Float4x4 &tm)
{
  glm::mat4 result;
  memcpy(glm::value_ptr(result), &tm.cols[0], sizeof(glm::mat4));
  return result;
}

static std::unique_ptr<Scene> scene;
static std::vector<std::string> animationList;


static struct {
  bool show_bones = true;
  bool show_arrows = true;
  bool show_mesh = true;
} visualisation_params;

static struct {
  float tolerance;
  float distance;
  float speed = 1;
  bool pause = false;
  bool loop = true;
  bool optimized = false;
} animation_params;

static int selectedNode = -1;

#include <filesystem>
static std::vector<std::string> scan_animations(const char *path)
{
  std::vector<std::string> animations;
  animations.push_back("None");
  for (auto &p : std::filesystem::recursive_directory_iterator(path))
  {
    auto filePath = p.path();
    if (p.is_regular_file() && filePath.extension() == ".fbx")
      animations.push_back(filePath.string());
  }
  return animations;
}

void resize_window_handler(const glm::vec2 &newSize)
{
  scene->userCamera.windowSize = newSize;
}

int IntersectRaySphere(glm::vec3 p, glm::vec3 d, glm::vec3 s_c, float s_r,  float &t, glm::vec3 &q) 
{
  glm::vec3 m = p - s_c; 
  float b = glm::dot(m, d); 
  float c = glm::dot(m, m) - s_r * s_r; 

  // Exit if r’s origin outside s (c > 0) and r pointing away from s (b > 0) 
  if (c > 0.0f && b > 0.0f) return 0; 
  float discr = b*b - c; 

  // A negative discriminant corresponds to ray missing sphere 
  if (discr < 0.0f) return 0; 

  // Ray now found to intersect sphere, compute smallest t value of intersection
  t = -b - std::sqrt(discr); 

  // If t is negative, ray started inside sphere so clamp t to zero 
  if (t < 0.0f) t = 0.0f; 
  q = p + t * d; 

  return 1;
}

// static glm::vec3 eye;
// static glm::vec3 dir;
// static glm::vec3 clickedPosViewport;
void mouse_click_handler(const SDL_MouseButtonEvent &e)
{
  if (e.button == SDL_BUTTON_RIGHT)
  {
    glm::vec3 right = scene->userCamera.transform[0];
    glm::vec3 up = scene->userCamera.transform[1];
    glm::vec3 forward = scene->userCamera.transform[2];
    glm::vec3 eye = scene->userCamera.transform[3];

    float viewportSizeY = 0.01f * std::tan(45.0f * DegToRad) * 2;
    float viewportSizeX = viewportSizeY * get_aspect_ratio();

    glm::vec3 leftUpCornerViewport = eye + 0.01f * forward - viewportSizeX / 2 * right +  viewportSizeY / 2 * up;
    glm::vec3 clickedPosViewport = leftUpCornerViewport
                                   + (e.x / scene->userCamera.windowSize[0]) * viewportSizeX * right 
                                   - (e.y / scene->userCamera.windowSize[1]) * viewportSizeY * up;
    glm::vec3 rayDir = glm::normalize(clickedPosViewport - eye);
    // dir = rayDir;

    float minDist = 1000;
    size_t intersectNodeIndex = -1;
    for (Character &character : scene->characters)
    {
      const auto &skeleton = *character.skeleton_;
      size_t nodeCount = skeleton.num_joints();
      for (size_t i = 0; i < nodeCount; ++i) 
      {
        glm::mat4 nodeTm = to_glm(character.models_[i]);      
        vec3 nodePos = character.transform * nodeTm[3];
        float distToNode;
        glm::vec3 intersectionPoint;
        float sphereRadius = 0.05;
        if (IntersectRaySphere(clickedPosViewport, rayDir, nodePos, sphereRadius, distToNode, intersectionPoint)) {
          if (distToNode < minDist) {
            minDist = distToNode;
            intersectNodeIndex = i;
          }
        }
      }
    }
    selectedNode = intersectNodeIndex;
  }
}

void game_init()
{
  animationList = scan_animations("resources/Animations");
  scene = std::make_unique<Scene>();
  scene->light.lightDirection = glm::normalize(glm::vec3(-1, -1, 0));
  scene->light.lightColor = glm::vec3(1.f);
  scene->light.ambient = glm::vec3(0.2f);

  scene->userCamera.projection = glm::perspective(90.f * DegToRad, get_aspect_ratio(), 0.01f, 500.f);

  ArcballCamera &cam = scene->userCamera.arcballCamera;
  cam.curZoom = cam.targetZoom = 0.5f;
  cam.maxdistance = 5.f;
  cam.distance = cam.curZoom * cam.maxdistance;
  cam.lerpStrength = 10.f;
  cam.mouseSensitivity = 0.5f;
  cam.wheelSensitivity = 0.05f;
  cam.targetPosition = glm::vec3(0.f, 1.f, 0.f);
  cam.targetRotation = cam.curRotation = glm::vec2(DegToRad * -90.f, DegToRad * -30.f);
  cam.rotationEnable = false;

  scene->userCamera.transform = calculate_transform(scene->userCamera.arcballCamera);

  input.onMouseButtonEvent += [](const SDL_MouseButtonEvent &e) { arccam_mouse_click_handler(e, scene->userCamera.arcballCamera); 
                                                                  mouse_click_handler(e);};
  input.onMouseMotionEvent += [](const SDL_MouseMotionEvent &e) { arccam_mouse_move_handler(e, scene->userCamera.arcballCamera); };
  input.onMouseWheelEvent += [](const SDL_MouseWheelEvent &e) { arccam_mouse_wheel_handler(e, scene->userCamera.arcballCamera); };
  input.onResizeEvent     += [](const glm::vec2 &newSize) { resize_window_handler(newSize); };

  auto material = make_material("character", "sources/shaders/character_vs.glsl", "sources/shaders/character_ps.glsl");
  std::fflush(stdout);
  material->set_property("mainTex", create_texture2d("resources/MotusMan_v55/MCG_diff.jpg"));
  
  SceneAsset sceneAsset = load_scene("resources/MotusMan_v55/MotusMan_v55.fbx", SceneAsset::LoadScene::Meshes | SceneAsset::LoadScene::Skeleton);
  Character &character = scene->characters.emplace_back(Character{
    glm::identity<glm::mat4>(),
    sceneAsset.meshes[0],
    std::move(material),
    sceneAsset.skeleton});

  character.controller.Reset();
  
  const int num_soa_joints = character.skeleton_->num_soa_joints();
  character.locals_.resize(num_soa_joints);
  const int num_joints = character.skeleton_->num_joints();
  character.models_.resize(num_joints);

  // Allocates a context that matches animation requirements.
  character.context_ = std::make_shared<ozz::animation::SamplingJob::Context>(num_joints);

  create_arrow_render();
  create_bone_render();
  std::fflush(stdout);
}

void render_imguizmo(ImGuizmo::OPERATION &mCurrentGizmoOperation, ImGuizmo::MODE &mCurrentGizmoMode)
{
  if (ImGui::Begin("gizmo window"))
  {
    if (ImGui::IsKeyPressed('Z'))
      mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
    if (ImGui::IsKeyPressed('E'))
      mCurrentGizmoOperation = ImGuizmo::ROTATE;
    if (ImGui::IsKeyPressed('R')) // r Key
      mCurrentGizmoOperation = ImGuizmo::SCALE;
    if (ImGui::RadioButton("Translate", mCurrentGizmoOperation == ImGuizmo::TRANSLATE))
      mCurrentGizmoOperation = ImGuizmo::TRANSLATE;
    ImGui::SameLine();
    if (ImGui::RadioButton("Rotate", mCurrentGizmoOperation == ImGuizmo::ROTATE))
      mCurrentGizmoOperation = ImGuizmo::ROTATE;
    ImGui::SameLine();
    if (ImGui::RadioButton("Scale", mCurrentGizmoOperation == ImGuizmo::SCALE))
      mCurrentGizmoOperation = ImGuizmo::SCALE;

    if (mCurrentGizmoOperation != ImGuizmo::SCALE)
    {
      if (ImGui::RadioButton("Local", mCurrentGizmoMode == ImGuizmo::LOCAL))
        mCurrentGizmoMode = ImGuizmo::LOCAL;
      ImGui::SameLine();
      if (ImGui::RadioButton("World", mCurrentGizmoMode == ImGuizmo::WORLD))
        mCurrentGizmoMode = ImGuizmo::WORLD;
    }

    ImGui::Text("Selected Node: %s", selectedNode == -1 ? "None" : scene->characters[0].skeleton_->joint_names()[selectedNode]);
  }
  ImGui::End();
}

static void playback_controller_inspector(PlaybackController &controller)
{
  ImGui::SliderFloat("progress", &controller.time_ratio_, 0.f, 1.f);
  ImGui::Checkbox("play/pause", &controller.play_);
  ImGui::Checkbox("is loop", &controller.loop_);
  ImGui::SliderFloat("speed", &controller.playback_speed_, -2.0f, 2.0f);

  if (ImGui::Button("reset"))
    controller.Reset();
}

static void optimizer_inspector(AnimationLayer& character)
{
  OptimizerController &optimizer = character.optimizer;
  ImGui::SliderFloat("Tolerance (mm)", &optimizer.tolerance, 0, 100);
  ImGui::SliderFloat("Distance (mm)", &optimizer.distance, 0, 1000);
  if (ImGui::Button("Apply optimize")) {
    AnimationPtr animation;
    SceneAsset sceneAsset;
    ExtraParameters extra = {.tolerance = optimizer.tolerance, .distance = optimizer.distance, .animation_info = &optimizer.animation_info};
    sceneAsset = load_scene(animationList[character.animationIndex].c_str(),
                                      SceneAsset::LoadScene::Skeleton | SceneAsset::LoadScene::Animation,
                                      &extra);
    if (!sceneAsset.animations.empty()) {
      animation = sceneAsset.animations[0];
    }
    optimizer.optimized = (optimizer.tolerance != 0.f || optimizer.distance != 0.f);
    character.animation = animation;
    character.controller.time_ratio_ = 0;
  }
  if (optimizer.optimized) {
    ImGui::Text("Optimized");
  } else {
    ImGui::Text("Non optimized");
  }
  ImGui::Text("Original: %llu", optimizer.animation_info.original);
  ImGui::Text("Optimized: %llu", optimizer.animation_info.optimized);
  ImGui::Text("Compressed: %llu", optimizer.animation_info.compressed);
}


static AnimationPtr animation_list_combo(SceneAsset::LoadScene animation_type, SkeletonPtr ref_pose, int *animationIndex, AnimationInfo *animation_info)
{
  static int item = 0;
  if (ImGui::ComboWithFilter("##anim", &item, animationList)) {
    AnimationPtr animation;
    if (item > 0)
    {
      ExtraParameters extra = {.animation_info = animation_info};
      SceneAsset sceneAsset = load_scene(animationList[item].c_str(),
                                         SceneAsset::LoadScene::Skeleton | SceneAsset::LoadScene::Animation, &extra);
      if (!sceneAsset.animations.empty()) {
        animation = sceneAsset.animations[0];
      }
      animation_params.optimized = false;
      if (!sceneAsset.animations.empty()) {
        *animationIndex = item;
        return sceneAsset.animations[0];
      }
    }
  }
  return nullptr;
}

void imgui_render()
{
  ImGuizmo::BeginFrame();
  for (Character &character : scene->characters)
  {
    if (ImGui::Begin("Visualisition"))
    {
      ImGui::Checkbox("Show mesh", &visualisation_params.show_mesh);
      ImGui::Checkbox("Show bones", &visualisation_params.show_bones);
      ImGui::Checkbox("Show arrows", &visualisation_params.show_arrows);
    }
    ImGui::End();

    static int item = 0;
    if (ImGui::Begin("Animation list"))
    {
      if (ImGui::Button("Add BlendTree1d")) {
        character.blendtrees.emplace_back();
      }
      for (size_t i = 0; i < character.blendtrees.size(); i++)
      {
        BlendTree1d &blendtree = character.blendtrees[i];
        std::string name = "Blend tree " + std::to_string(i + 1);
        if (ImGui::TreeNode(name.c_str()))
        {
          ImGui::SliderFloat("weight", &blendtree.weight, -1.f, 1.f);
          if (ImGui::Button("Play all")) {
             for (int j = 0; j < blendtree.nLayers; ++j) {
              if (blendtree.layersId[j] != -1) {
                AnimationLayer &layer = character.layers[blendtree.layersId[j]];
                layer.controller.play_ = true;
              }
             }
          }
          if (ImGui::Button("Stop all")) {
             for (int j = 0; j < blendtree.nLayers; ++j) {
              if (blendtree.layersId[j] != -1) {
                AnimationLayer &layer = character.layers[blendtree.layersId[j]];
                layer.controller.play_ = false;
              }
             }
          }
          for (int j = 0; j < blendtree.nLayers; ++j) {
            if (blendtree.layersId[j] != -1) {
              AnimationLayer &layer = character.layers[blendtree.layersId[j]];
              if (layer.visability) {
                ImGui::PushID(j);
                ImGui::Text("name: %s", layer.animation->name());
                // ImGui::SameLine();
                // if (ImGui::Button("disconnect")) {
                //   layer.owned_by_blendtree = false;
                //   blendtree.layers[j] = -1;
                //   std::swap(blendtree.layers[j], blendtree.layers[blendtree.nLayers - 1]);
                //   --blendtree.nLayers;
                //   --j;
                //   ImGui::PopID();
                //   continue;
                // }
                ImGui::Text("duration: %f", layer.animation->duration());
                ImGui::SliderFloat("weight", &layer.weight, 0.f, 1.f);
                // ImGui::Text("%s", layer.isAdditive ? "is additive" : "not additive");

                if (ImGui::TreeNode("controller"))
                {
                  playback_controller_inspector(layer.controller);
                  ImGui::TreePop();
                }
                if (ImGui::TreeNode("optimizer"))
                {
                  optimizer_inspector(layer);
                  ImGui::TreePop();
                }
                ImGui::PopID();
              }
            }
          }
          ImGui::TreePop();
        }
      }
      ImGui::NewLine(); ImGui::NewLine(); ImGui::NewLine(); ImGui::NewLine(); ImGui::NewLine();

      if (ImGui::Button("Add layer"))
        ImGui::OpenPopup("Add layer to play");
      if (ImGui::BeginPopup("Add layer to play"))
      {
        int animationIndex = -1;
        AnimationInfo animation_info;
        if (AnimationPtr animation = animation_list_combo(SceneAsset::LoadScene::Animation, character.skeleton_,  &animationIndex, &animation_info))
        {
          character.layers.emplace_back(character.skeleton_, animation, animationIndex, animation_info);
          ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
      }
      for (size_t i = 0; i < character.layers.size(); i++)
      {
        AnimationLayer &layer = character.layers[i];
        if (layer.visability && !layer.owned_by_blendtree) {
          ImGui::PushID(i);
          ImGui::Text("name: %s", layer.animation->name());
          ImGui::SameLine();
          // if (ImGui::Button("Delete layer")) {
          //   layer.weight = 0;
          //   layer.visability = false;
          // }
          ImGui::SameLine();
          if (ImGui::Button("Connect to blend tree"))
            ImGui::OpenPopup("Connect to blend tree popup");
          if (ImGui::BeginPopup("Connect to blend tree popup"))
          {
            std::vector<std::string> blendtrees(character.blendtrees.size() + 1);
            blendtrees[0] = "None";
            for (size_t i = 1; i < character.blendtrees.size() + 1; i++)
              blendtrees[i] = "BlendTree " + std::to_string(i);
            static int item = 0;
            if (ImGui::ComboWithFilter("", &item, blendtrees))
            {
              if (item > 0)
              {
                size_t ind = character.blendtrees[item - 1].nLayers;
                if (ind < 3) {
                  character.blendtrees[item - 1].layersId[ind] = i;
                  layer.owned_by_blendtree = true;
                  ++character.blendtrees[item - 1].nLayers;
                }
              }
              ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
          }
          ImGui::Text("duration: %f", layer.animation->duration());
          ImGui::SliderFloat("weight", &layer.weight, 0.f, 1.f);
          // ImGui::Text("%s", layer.isAdditive ? "is additive" : "not additive");

          if (ImGui::TreeNode("controller"))
          {
            playback_controller_inspector(layer.controller);
            ImGui::TreePop();
          }
          if (ImGui::TreeNode("optimizer"))
          {
            optimizer_inspector(layer);
            ImGui::TreePop();
          }
          ImGui::PopID();
        }
      }
    }
    ImGui::End();

    static ImGuizmo::OPERATION mCurrentGizmoOperation(ImGuizmo::TRANSLATE);
    static ImGuizmo::MODE mCurrentGizmoMode(ImGuizmo::WORLD);
    render_imguizmo(mCurrentGizmoOperation, mCurrentGizmoMode);

    const glm::mat4 &projection = scene->userCamera.projection;
    const glm::mat4 &transform = scene->userCamera.transform;
    mat4 cameraView = inverse(transform);
    ImGuiIO &io = ImGui::GetIO();
    ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);
    
    glm::mat4 globNodeTm = character.transform;

    ImGuizmo::Manipulate(glm::value_ptr(cameraView), glm::value_ptr(projection), mCurrentGizmoOperation, mCurrentGizmoMode,
                         glm::value_ptr(globNodeTm));

    // int parent = skeleton.ref->parent[idx];
    // character.skeleton.localTm[idx] = glm::inverse(parent >= 0 ? character.skeleton.globalTm[parent] : glm::mat4(1.f)) * globNodeTm;
    character.transform = globNodeTm;
    break;
  }
}

void game_update()
{
  float dt = get_delta_time();
  arcball_camera_update(
    scene->userCamera.arcballCamera,
    scene->userCamera.transform,
    dt);
  for (Character &character : scene->characters)
  {
    if (!character.layers.empty()) {
      for (BlendTree1d &blendtree : character.blendtrees)
      {
        float weights[3];
        if (blendtree.weight < 0) {
          weights[2] = 0;
          weights[1] = blendtree.weight - (-1);
          weights[0] = 1 - weights[1];
        } else {
          weights[0] = 0;
          weights[2] = blendtree.weight;
          weights[1] = 1 - weights[2]; 
        }
        for (size_t i = 0; i < blendtree.nLayers; ++i) {
          character.layers[blendtree.layersId[i]].weight = weights[i];
        }
        float avg_duration = 0; // не до конца понял почему именно взвешанная сумма, а не, например, минимум из двух, у которых веса > 0
                                // а это посмотрел как одногруппники сделали и так же сделал, вроде звучит как будто правильно,
                                // может я тут не правильно списал даже, в спешке смотрел просто 
                                // но все равно приходится вручную находить тайминги, когда ноги в одинаковых направлениях, не понял как это автоматически сделать
        for (size_t i = 0; i < blendtree.nLayers; ++i) {
          avg_duration += character.layers[blendtree.layersId[i]].animation->duration() * weights[i];
        }
        for (size_t i = 0; i < blendtree.nLayers; ++i) {
          character.layers[blendtree.layersId[i]].controller.playback_speed_ = character.layers[blendtree.layersId[i]].animation->duration() / avg_duration;
        }
      }
      for (AnimationLayer &layer : character.layers)
      {
        layer.controller.Update(layer.animation, dt);

        // Samples optimized animation at t = animation_time_.
        ozz::animation::SamplingJob sampling_job;
        sampling_job.animation = layer.animation.get();
        sampling_job.context = layer.context.get();
        sampling_job.ratio = layer.controller.time_ratio_;
        sampling_job.output = ozz::make_span(layer.locals);
        if (!sampling_job.Run())
        {
          debug_error("sampling_job failed");
        }
      }

      // Prepares blending layers.
      int numLayer = character.layers.size();
      std::vector<ozz::animation::BlendingJob::Layer> layers, additive;

      for (int i = 0; i < numLayer; ++i)
      {
        ozz::animation::BlendingJob::Layer layer;
        layer.transform = ozz::make_span(character.layers[i].locals);
        layer.weight = character.layers[i].weight;
        if (!character.layers[i].isAdditive)
          layers.push_back(layer);
        else
          additive.push_back(layer);
      }

      // Setups blending job.
      ozz::animation::BlendingJob blend_job;
      blend_job.threshold = 0.1;
      blend_job.layers = ozz::make_span(layers);
      blend_job.additive_layers = ozz::make_span(additive);
      blend_job.rest_pose = character.skeleton_->joint_rest_poses();
      blend_job.output = ozz::make_span(character.locals_);

      // Blends.
      if (!blend_job.Run())
      {
        debug_error("blend_job failed");
        continue;
      }
    }
    else if (character.currentAnimation)
    {
      character.controller.Update(character.currentAnimation, dt);
      // character.animTime += animation_params.pause ? 0 : dt * animation_params.speed;
      // if (character.animTime >= character.currentAnimation->duration()) {
      //   if (animation_params.loop) {
      //     character.animTime = 0;
      //   } else {
      //     character.animTime = character.currentAnimation->duration();
      //   }
      // }

      // Samples optimized animation at t = animation_time_.
      ozz::animation::SamplingJob sampling_job;
      sampling_job.animation = character.currentAnimation.get();
      sampling_job.context = character.context_.get();
      // sampling_job.ratio = character.animTime / character.currentAnimation->duration();
      sampling_job.ratio = character.controller.time_ratio_;
      sampling_job.output = ozz::make_span(character.locals_);
      if (!sampling_job.Run())
      {
        continue;
      }
    }
    else
    {
      auto restPose = character.skeleton_->joint_rest_poses();
      std::copy(restPose.begin(), restPose.end(), character.locals_.begin());
    }
    ozz::animation::LocalToModelJob ltm_job;
    ltm_job.skeleton = character.skeleton_.get();
    ltm_job.input = ozz::make_span(character.locals_);
    ltm_job.output = ozz::make_span(character.models_);
    if (!ltm_job.Run())
    {
      continue;
    }
  }
}

void render_character(const Character &character, const mat4 &cameraProjView, vec3 cameraPosition, const DirectionLight &light)
{
  if (visualisation_params.show_mesh) {
    const Material &material = *character.material;
    const Shader &shader = material.get_shader();

    shader.use();
    material.bind_uniforms_to_shader();
    shader.set_mat4x4("Transform", character.transform);
    shader.set_mat4x4("ViewProjection", cameraProjView);
    shader.set_vec3("CameraPosition", cameraPosition);
    shader.set_vec3("LightDirection", glm::normalize(light.lightDirection));
    shader.set_vec3("AmbientLight", light.ambient);
    shader.set_vec3("SunLight", light.lightColor);

    size_t boneNumber = character.mesh->bones.size();
    std::vector<mat4> bones(boneNumber);

    const auto &skeleton = *character.skeleton_;
    size_t nodeCount = skeleton.num_joints();
    for (size_t i = 0; i < nodeCount; i++)
    {
      auto it = character.mesh->bonesMap.find(skeleton.joint_names()[i]);
      if (it != character.mesh->bonesMap.end())
      {
        int boneIdx = it->second;
        bones[boneIdx] = to_glm(character.models_[i]) * character.mesh->bones[boneIdx].invBindPose;
      }
    }
    shader.set_mat4x4("Bones", bones);

    render(character.mesh);
  }
  
  const auto &skeleton = *character.skeleton_;
  size_t nodeCount = skeleton.num_joints();
  for (size_t i = 0; i < nodeCount; i++)
  { 
    float distanceToParent = 0.0f;

    int parentId = skeleton.joint_parents()[i];
    glm::mat4 boneTm   =  character.transform * to_glm(character.models_[i]);  
    if (parentId != -1 && parentId != 0 && parentId != 1) {
      glm::mat4 parentTm =  character.transform * to_glm(character.models_[parentId]);      
      vec3 parentPos = parentTm[3];
      vec3 bonePos = boneTm[3];
      distanceToParent = glm::length(bonePos - parentPos);
      if (visualisation_params.show_bones) {
        constexpr vec3 darkGreenColor = vec3(46.0 / 255, 142.0 / 255, 16.0 / 255);
        draw_bone(parentPos, bonePos, darkGreenColor, 0.1f);
      }
    }

    if (visualisation_params.show_arrows) {
      const float arrowLength = (distanceToParent + 0.1) / 5.0;
      constexpr float arrowSize = 0.004f;
      draw_arrow(boneTm, vec3(0), vec3(arrowLength, 0, 0), vec3(1, 0, 0), arrowSize);
      draw_arrow(boneTm, vec3(0), vec3(0, arrowLength, 0), vec3(0, 1, 0), arrowSize);
      draw_arrow(boneTm, vec3(0), vec3(0, 0, arrowLength), vec3(0, 0, 1), arrowSize);
    }
  }
}

void game_render()
{
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);
  const float grayColor = 0.3f;
  glClearColor(grayColor, grayColor, grayColor, 1.f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


  const glm::mat4 &projection = scene->userCamera.projection;
  const glm::mat4 &transform = scene->userCamera.transform;
  glm::mat4 projView = projection * inverse(transform);

  for (const Character &character : scene->characters)
    render_character(character, projView, glm::vec3(transform[3]), scene->light);


  render_bones(projView, glm::vec3(transform[3]), scene->light);
  render_arrows(projView, glm::vec3(transform[3]), scene->light);
}

void close_game()
{
  scene.reset();
}