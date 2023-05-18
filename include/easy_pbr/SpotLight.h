#pragma once

#include <memory>

#include "easy_gl/Shader.h"
#include "easy_gl/GBuffer.h"

#include <Eigen/Geometry>

#include "easy_pbr/Camera.h"

#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include <configuru.hpp>

//to his the fact that it is inheriting from Camera and both inherit from enabled_shared_from https://www.codeproject.com/Articles/286304/Solution-for-multiple-enable-shared-from-this-in-i
#include "shared_ptr/EnableSharedFromThis.h"
#include "shared_ptr/SmartPtrBuilder.h"

namespace easy_pbr{

class MeshGL;

// class SpotLight : public std::enable_shared_from_this<SpotLight>, public Camera
class SpotLight : public Camera, public Generic::EnableSharedFromThis< SpotLight >
{
public:
    // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SpotLight(const configuru::Config& config, const configuru::Config& default_config);

    // void render_to_shadow_map(const MeshCore& mesh);
    void set_power_for_point(const Eigen::Vector3f& point, const float power);
    void render_mesh_to_shadow_map(std::shared_ptr<MeshGL>& mesh);
    void render_points_to_shadow_map(std::shared_ptr<MeshGL>& mesh);
    void clear_shadow_map();
    void blur_shadow_map(const int nr_iters);
    void set_shadow_map_resolution(const int shadow_map_resolution);
    int shadow_map_resolution();
    bool has_shadow_map();
    gl::Texture2D& get_shadow_map_ref();
    gl::GBuffer& get_shadow_map_fbo_ref();


    //inherited movement function from camera. Whenever we move a light, it's shadow map is defined as dirty
    void set_model_matrix(const Eigen::Affine3f & delta);
    void set_lookat(const Eigen::Vector3f& lookat); //updates the orientation according to the up vector so that it points towards lookat
    void set_position(const Eigen::Vector3f& pos); //updates the orientation according to the up vector so that it keeps pointing towards lookat
    void set_quat(const Eigen::Vector4f& quat); //sets the quaternion of the model matrix (tf_world_cam)
    void set_up(const Eigen::Vector3f& up);
    void set_dist_to_lookat(const float dist); //sets the lookat at a certain distance along the negative z axis
    void transform_model_matrix(const Eigen::Affine3f & delta);
    void move_cam_and_lookat(const Eigen::Vector3f& pos); //moves the camera together with the lookat point
    void dolly(const Eigen::Vector3f& dv); //moves the camera along a certain displacement vector dv expressed in world coordinates
    void push_away(const float s); //moves the camera closer or further from the lookup point. A 's' values of 1 means no movement s>1 means going further and s<1 means closer
    void push_away_by_dist(const float new_dist); //pueshes the camera backwards or forwards until the distance to lookat point matches the new_dist
    void orbit(const Eigen::Quaternionf& q); //Orbit around the m_lookat by an amount specified by q
    void orbit_x(const float angle_degrees); //orbit around the x axis of the world a certain amount of degrees
    void orbit_y(const float angle_degrees); //orbit around the y axis of the world a certain amount of degrees
    void orbit_z(const float angle_degrees); //orbit around the z axis of the world a certain amount of degrees
    void orbit_axis_angle(const Eigen::Vector3f& axis, const float angle_degrees); //orbit around the chosen axis of the world a certain amount of degrees
    void rotate(const Eigen::Quaternionf& q); //rotates around the central camera position by a quaternion q
    void rotate_axis_angle(const Eigen::Vector3f& axis, const float angle_degrees); //same as rotate but using a axis_angle
    void from_frame(const Frame& frame); // initialized the camera to have the parameters of the frame
    


    void print_ptr();

    float m_power;
    Eigen::Vector3f m_color;
    float m_penumbra_size;
    bool m_create_shadow;
    bool m_is_shadowmap_dirty; //whenever the light moves, the shadowmap is set as dirty

private:

    void init_params(const configuru::Config& config_file, const configuru::Config& default_config);
    void init_opengl();
    void blur_tex(gl::Texture2D& tex_in, gl::Texture2D& tex_out, gl::Texture2D&tex_tmp, const int nr_iters);

    gl::Shader m_shadow_map_shader;
    gl::Shader m_blur_shader;
    gl::GBuffer m_shadow_map_fbo; //fbo that contains only depth maps for usage as a shadow map
    int m_shadow_map_resolution;
    std::shared_ptr<MeshGL> m_fullscreen_quad;
    

};

} //namespace easy_pbr
