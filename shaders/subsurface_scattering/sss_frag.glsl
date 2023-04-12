#version 330 core
#extension GL_ARB_separate_shader_objects : require
#extension GL_ARB_explicit_attrib_location : require

//following mostly https://github.com/HanetakaChou/Subsurface-Scattering-Disney/blob/master/Shaders/subsurface_scattering_disney_blur.hlsli

//in
layout(location=1) in vec2 uv_in;

//out
layout(location = 0) out vec4 blurred_output;

// uniform sampler2D img;
uniform sampler2D composed_diffuse_gtex;
//TODO need
uniform sampler2D depth_tex;
// uniform sampler2D ao_and_needs_sss_gtex;
uniform sampler2D metalness_and_roughness_and_sss_strength_tex;

// uniform bool horizontal;
uniform float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

//TODO to be filled
uniform mat4 currProj;
uniform float projection_a; //for calculating position from depth according to the formula at the bottom of article https://mynameismjp.wordpress.com/2010/09/05/position-from-depth-3/
uniform float projection_b;
// uniform float fov_y;
uniform float fov_x;
uniform bool is_ortho;
uniform bool horizontal;
// uniform float cam_near;
// uniform float cam_far;
uniform float sss_width;

// vec4 kernel[] = {
// 	vec4(0.530605, 0.613514, 0.739601, 0),
// 	vec4(0.000973794, 1.11862e-005, 9.43437e-007, -3),
// 	vec4(0.00333804, 7.85443e-005, 1.2945e-005, -2.52083),
// 	vec4(0.00500364, 0.00020094, 5.28848e-005, -2.08333),
// 	vec4(0.00700976, 0.00049366, 0.000151938, -1.6875),
// 	vec4(0.0094389, 0.00139119, 0.000416598, -1.33333),
// 	vec4(0.0128496, 0.00356329, 0.00132016, -1.02083),
// 	vec4(0.017924, 0.00711691, 0.00347194, -0.75),
// 	vec4(0.0263642, 0.0119715, 0.00684598, -0.520833),
// 	vec4(0.0410172, 0.0199899, 0.0118481, -0.333333),
// 	vec4(0.0493588, 0.0367726, 0.0219485, -0.1875),
// 	vec4(0.0402784, 0.0657244, 0.04631, -0.0833333),
// 	vec4(0.0211412, 0.0459286, 0.0378196, -0.0208333),
// 	vec4(0.0211412, 0.0459286, 0.0378196, 0.0208333),
// 	vec4(0.0402784, 0.0657244, 0.04631, 0.0833333),
// 	vec4(0.0493588, 0.0367726, 0.0219485, 0.1875),
// 	vec4(0.0410172, 0.0199899, 0.0118481, 0.333333),
// 	vec4(0.0263642, 0.0119715, 0.00684598, 0.520833),
// 	vec4(0.017924, 0.00711691, 0.00347194, 0.75),
// 	vec4(0.0128496, 0.00356329, 0.00132016, 1.02083),
// 	vec4(0.0094389, 0.00139119, 0.000416598, 1.33333),
// 	vec4(0.00700976, 0.00049366, 0.000151938, 1.6875),
// 	vec4(0.00500364, 0.00020094, 5.28848e-005, 2.08333),
// 	vec4(0.00333804, 7.85443e-005, 1.2945e-005, 2.52083),
// 	vec4(0.000973794, 1.11862e-005, 9.43437e-007, 3)
// };
uniform vec4 kernel[25] = vec4[] (
    vec4(0.530605, 0.613514, 0.739601, 0),
	vec4(0.000973794, 1.11862e-005, 9.43437e-007, -3),
	vec4(0.00333804, 7.85443e-005, 1.2945e-005, -2.52083),
	vec4(0.00500364, 0.00020094, 5.28848e-005, -2.08333),
	vec4(0.00700976, 0.00049366, 0.000151938, -1.6875),
	vec4(0.0094389, 0.00139119, 0.000416598, -1.33333),
	vec4(0.0128496, 0.00356329, 0.00132016, -1.02083),
	vec4(0.017924, 0.00711691, 0.00347194, -0.75),
	vec4(0.0263642, 0.0119715, 0.00684598, -0.520833),
	vec4(0.0410172, 0.0199899, 0.0118481, -0.333333),
	vec4(0.0493588, 0.0367726, 0.0219485, -0.1875),
	vec4(0.0402784, 0.0657244, 0.04631, -0.0833333),
	vec4(0.0211412, 0.0459286, 0.0378196, -0.0208333),
	vec4(0.0211412, 0.0459286, 0.0378196, 0.0208333),
	vec4(0.0402784, 0.0657244, 0.04631, 0.0833333),
	vec4(0.0493588, 0.0367726, 0.0219485, 0.1875),
	vec4(0.0410172, 0.0199899, 0.0118481, 0.333333),
	vec4(0.0263642, 0.0119715, 0.00684598, 0.520833),
	vec4(0.017924, 0.00711691, 0.00347194, 0.75),
	vec4(0.0128496, 0.00356329, 0.00132016, 1.02083),
	vec4(0.0094389, 0.00139119, 0.000416598, 1.33333),
	vec4(0.00700976, 0.00049366, 0.000151938, 1.6875),
	vec4(0.00500364, 0.00020094, 5.28848e-005, 2.08333),
	vec4(0.00333804, 7.85443e-005, 1.2945e-005, 2.52083),
	vec4(0.000973794, 1.11862e-005, 9.43437e-007, 3)
);


float linear_depth(float depth_sample){
    // depth_sample = 2.0 * depth_sample - 1.0;
    // float z_linear = 2.0 * z_near * z_far / (z_far + z_near - depth_sample * (z_far - z_near));
    // return z_linear;

    // according to the formula at the bottom of article https://mynameismjp.wordpress.com/2010/09/05/position-from-depth-3/
    // float ProjectionA = z_far / (z_far - z_near);
    // float ProjectionB = (-z_far * z_near) / (z_far - z_near);
    
    float linearDepth;
    if (is_ortho){
        linearDepth= depth_sample;
    }else{
        linearDepth= projection_b / (depth_sample - projection_a);
    }

    return linearDepth;

}


// float linear_depth(float z){
// 	float n = cam_near;
// 	float f = cam_far;
// 	return (n * f) / (f - z * (f - n));
// }

void main(){

    //https://github.com/DoerriesT/Separable-Subsurface-Scattering-Demo/blob/7f98026f06d10950f218849dffde45667f884408/SubsurfaceScattering/src/vulkan/Renderer.cpp#L531
    // pushConsts.texelSize = 1.0f / glm::vec2(m_width, m_height);
    // pushConsts.dir = glm::vec2(1.0f, 0.0f);
    // pushConsts.sssWidth = sssWidth * 1.0f / tanf(fovy * 0.5f) * (m_height / static_cast<float>(m_width)); 
    // ivec2 tex_size= textureSize(composed_diffuse_gtex, 0);
    // float width=tex_size.x;
    // float height=tex_size.y;
    // vec2 texelSize = 1.0 / tex_size; // gets size of single texel
    // vec2 dir=vec2(1.0, 0.0);
    vec2 dir;
    if (horizontal){
        dir=vec2(1.0, 0.0);   
    }else{
        dir=vec2(0.0, 1.0);  
    }
    // float sss_width=0.002;
    // float sss_width=0.02;
    // float sss_width_world=sss_width* 1.0/ tan(fov_y * 0.5) * (height / width);
    float sss_width_world=sss_width;
    // float maxDepthDiff = 0.1;




    // vec3 color = subsurface_scattering_disney_blur(scatteringDistance, worldScale, pixelsPerSample, sampleBudget, uv_in);

    float depth_val = texture(depth_tex, uv_in).x;
    float linear_depth_val=linear_depth(depth_val);

    // float rayRadiusUV = 0.5 * sss_width_screen / linear_depth_val;
    float rayRadiusUV = sss_width_world / (linear_depth_val *fov_x) ;


    // early out if kernel footprint is less than a pixel
	// if (rayRadiusUV <= texelSize.x){
	// 	// imageStore(uResultImage, ivec2(gl_GlobalInvocationID.xy), colorM);
    //     blurred_output=vec4(colorM);
	// 	return;
	// }

    vec4 colorM = texture(composed_diffuse_gtex, uv_in);
    float needs_sss_f=texture(metalness_and_roughness_and_sss_strength_tex, uv_in).z;
    bool needs_sss=needs_sss_f>0.5;
    if(!needs_sss){
        blurred_output=vec4(colorM);
        return;
    }


    // calculate the final step to fetch the surrounding pixels:
	vec2 finalStep = rayRadiusUV * dir;
	finalStep *= 1.0 / 3.0; // divide by 3 as the kernels range from -3 to 3
	
	// accumulate the center sample:
	vec4 colorBlurred = colorM;
	colorBlurred.rgb *= kernel[0].rgb;
	
	// vec2 texCoord = (vec2(gl_GlobalInvocationID.xy) + vec2(0.5)) * uPushConsts.texelSize;
    vec2 texCoord=uv_in;
	
	// // accumulate the other samples:
	// for (int i = 1; i < 25; ++i){
	// 	// fetch color and depth for current sample:
	// 	vec2 offset = texCoord + kernel[i].a * finalStep;
	// 	vec4 color = textureLod(composed_diffuse_gtex, offset, 0.0);
	// 	float depth = linear_depth(textureLod(depth_tex, offset, 0.0).x);
    //     float needs_sss_f_offset=texture(metalness_and_roughness_and_sss_strength_tex, offset).z;
		
	// 	// lerp back to center sample if depth difference too big
	// 	float alpha = min(distance(depth, linear_depth_val) / maxDepthDiff, maxDepthDiff);
		
	// 	// reject sample if it isnt tagged as SSS
	// 	alpha *= 1.0 - needs_sss_f_offset;
		
	// 	color.rgb = mix(color.rgb, colorM.rgb, alpha);
		
	// 	// accumulate:
	// 	colorBlurred.rgb += kernel[i].rgb * color.rgb;
	// }



    // accumulate the other samples:
	for (int i = 1; i < 25; ++i){
		// fetch color and depth for current sample:
		vec2 offset = texCoord + kernel[i].a * finalStep;
		vec4 color_offset = textureLod(composed_diffuse_gtex, offset, 0.0);
		float linear_depth_offset = linear_depth(textureLod(depth_tex, offset, 0.0).x);
        float needs_sss_f_offset=texture(metalness_and_roughness_and_sss_strength_tex, offset).z;
		
		// lerp back to center sample if depth difference too big
		// float alpha = min(distance(linear_depth_offset, linear_depth_val) / maxDepthDiff, maxDepthDiff);
		// reject sample if it isnt tagged as SSS
		// alpha *= 1.0 - needs_sss_f_offset;
		// color_offset.rgb = mix(color_offset.rgb, colorM.rgb, alpha);
		
		// accumulate:
		colorBlurred.rgb += kernel[i].rgb * color_offset.rgb;
	}

    
    // blurred_output=vec4(vec3(color), 1.0);
    blurred_output=vec4(colorBlurred);
    // blurred_output=vec4(linear_depth_val);
    // blurred_output=vec4(linear_depth_val);
}






























// #version 330 core
// #extension GL_ARB_separate_shader_objects : require
// #extension GL_ARB_explicit_attrib_location : require

// //following mostly https://github.com/HanetakaChou/Subsurface-Scattering-Disney/blob/master/Shaders/subsurface_scattering_disney_blur.hlsli

// //in
// layout(location=1) in vec2 uv_in;

// //out
// layout(location = 0) out vec4 blurred_output;

// // uniform sampler2D img;
// uniform sampler2D composed_diffuse_gtex;
// //TODO need
// uniform sampler2D depth_tex;
// // uniform sampler2D ao_and_needs_sss_gtex;
// uniform sampler2D metalness_and_roughness_and_sss_strength_tex;
// //from https://github.com/HanetakaChou/Subsurface-Scattering-Disney/blob/b4debf1d277a555d8261ef7f474a8610bf43e1b5/Shaders/Support/Main.hlsli#L372
// uniform sampler2D g_total_diffuse_reflectance_pre_scatter_multiply_form_factor_texture; //seems to be the actual composed image we have
// //from https://github.com/HanetakaChou/Subsurface-Scattering-Disney/blob/b4debf1d277a555d8261ef7f474a8610bf43e1b5/Shaders/Support/Main.hlsli#L368
// uniform sampler2D g_albedo_texture; //seems to be just the albedo of the mesh, before any ilumination

// // uniform bool horizontal;
// uniform float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

// //TODO to be filled
// uniform mat4 currProj;
// // uniform vec3 scatteringDistance;
// // uniform  float padding_scatteringDistance;
// // uniform float worldScale;
// // uniform float postscatterEnabled;
// // uniform int sampleBudget;
// // uniform int pixelsPerSample;


// #define PI 3.141592653589793238462643
// #define FLT_MIN 1.175494351e-38
// #define LOG2_E 1.44269504088896340736
// #define SSS_MIN_PIXELS_PER_SAMPLE 4
// #define SSS_MAX_SAMPLE_BUDGET 256

// //bitfieldReverse is only available after opengl 4.0 so we get a reference implementation from: https://developer.download.nvidia.com/cg/bitfieldReverse.html
// int bitfieldReverse(int x){
//   int res = 0;
//   int i, shift, mask;

//   for(i = 0; i < 32; i++) {
//     mask = 1 << i;
//     shift = 32 - 2*i - 1;
//     mask &= x;
//     mask = (shift > 0) ? mask << shift : mask >> -shift;
//     res |= mask;
//   }

//   return res;
// }


// vec2 hammersley_2d(int sample_index, int sample_count){
//     // "7.4.1 Hammersley and Halton Sequences" of PBR Book
//     // UE: [Hammersley](https://github.com/EpicGames/UnrealEngine/blob/4.27/Engine/Shaders/Private/MonteCarlo.ush#L34)
//     // U3D: [Hammersley2d](https://github.com/Unity-Technologies/Graphics/blob/v10.8.0/com.unity.render-pipelines.core/ShaderLibrary/Sampling/Hammersley.hlsl#L415)

//     float xi_1 = float(sample_index) / float(sample_count);
//     float xi_2 = bitfieldReverse(sample_index) * (1.0 / 4294967296.0);

//     return vec2(xi_1, xi_2);
// }

// float diffusion_profile_sample_r(float d, float cdf){
// 	float u = 1 - cdf; // Convert CDF to CCDF

// 	float g = 1 + (4 * u) * (2 * u + sqrt(1 + (4 * u) * u));

// 	// g^(-1/3)
// 	float n = exp2(log2(g) * (-1.0 / 3.0));
// 	// g^(+1/3)
// 	float p = (g * n) * n;
// 	// 1 + g^(+1/3) + g^(-1/3)
// 	float c = 1 + p + n;
// 	// 3 * Log[4 * u]
// 	float b = (3 / LOG2_E * 2) + (3 / LOG2_E) * log2(u);
// 	// 3 * Log[c / (4 * u)]
// 	float x = (3 / LOG2_E) * log2(c) - b;

// 	// x = S * r
// 	// r = x * rcpS = x * d
// 	float r = x * d;
// 	return r;
// }

// float diffusion_profile_evaluate_cdf(float d, float r){
// 	// x = S * r = (1.0 / d) * r
// 	// exp_13 = Exp[-x/3]
// 	float exp_13 = exp2(((LOG2_E * (-1.0 / 3.0)) * r) * (1.0 / d));
// 	// exp_1  = Exp[-x] = exp_13 * exp_13 * exp_13
// 	// exp_sum = -0.25 * exp_1 - 0.75 * exp_13 =  exp_13 * (-0.75 - 0.25 * exp_13 * exp_13)
// 	float exp_sum = exp_13 * (-0.75 - 0.25 * exp_13 * exp_13);
// 	// 1 - 0.75 * Exp[-S * r / 3]  - 0.25 * Exp[-S * r])
// 	float cdf = 1.0 + exp_sum;
// 	return cdf;
// }

// float diffusion_profile_evaluate_rcp_pdf(float d, float r){
// 	// x = S * r = (1.0 / d) * r
// 	// exp_13 = Exp[-x/3]
// 	float exp_13 = exp2(((LOG2_E * (-1.0 / 3.0)) * r) * (1.0 / d));
// 	// exp_1  = Exp[-x] = exp_13 * exp_13 * exp_13
// 	// exp_sum = exp_1 + exp_13 =  exp_13 * (1 + exp_13 * exp_13)
// 	float exp_sum = exp_13 * (1 + exp_13 * exp_13);
// 	// rcpExp = (1.0 / exp_sum)
// 	float rcpExp = (1.0 / exp_sum);

// 	// rcpS = d
// 	// (8 * PI) / S / (Exp[-S * r / 3] + Exp[-S * r])
// 	float rcp_pdf = (8.0 * PI) * d * rcpExp;
// 	return rcp_pdf;
// }

// vec3 diffusion_profile_evaluate_pdf(vec3 S, float r){
// 	// Exp[-s * r / 3]
// 	vec3 exp_13 = exp2(((LOG2_E * (-1.0 / 3.0)) * r) * S);
// 	// Exp[-s * r / 3] + Exp[-S * r]
// 	vec3 exp_sum = exp_13 * (1 + exp_13 * exp_13);
// 	// S / (8 * PI) * (Exp[-S * r / 3] + Exp[-S * r])
// 	vec3 pdf = S / (8.0 * PI) * exp_sum;
// 	return pdf;
// }


// //from https://github.com/HanetakaChou/Subsurface-Scattering-Disney/blob/master/Shaders/subsurface_scattering_texturing_mode.hlsli
// vec3 subsurface_scattering_total_diffuse_reflectance_pre_scatter_from_albedo(bool is_post_scatter_texturing_mode, vec3 albedo){
// 	if (!is_post_scatter_texturing_mode){
// 		return sqrt(albedo);
// 	}else{
// 		return vec3(1.0);
// 	}
// }
// vec3 subsurface_scattering_total_diffuse_reflectance_post_scatter_from_albedo(bool is_post_scatter_texturing_mode, vec3 albedo){
// 	if (!is_post_scatter_texturing_mode){
// 		return sqrt(albedo);
// 	}else{
// 		return albedo;
// 	}
// }

// //from https://github.com/HanetakaChou/Subsurface-Scattering-Disney/blob/master/Shaders/Support/SSS_Blur.hlsli
// vec3 SSS_TOTAL_DIFFUSE_REFLECTANCE_PRE_SCATTER_MULTIPLY_FORM_FACTOR_SOURCE(vec2 pixelCoord){
// 	// NOTE: use SampleLevel rather than Sample
// 	// warning X3595: gradient instruction used in a loop with varying iteration; partial derivatives may have undefined value
// 	// vec3 total_diffuse_reflectance_pre_scatter_multiply_form_factor = g_total_diffuse_reflectance_pre_scatter_multiply_form_factor_texture.SampleLevel(PointSampler, pixelCoord, 0).rgb;
// 	vec3 total_diffuse_reflectance_pre_scatter_multiply_form_factor = texture(g_total_diffuse_reflectance_pre_scatter_multiply_form_factor_texture, pixelCoord).rgb;
// 	return total_diffuse_reflectance_pre_scatter_multiply_form_factor;
// }
// // vec3 SSS_TOTAL_DIFFUSE_REFLECTANCE_POST_SCATTER_SOURCE(vec2 pixelCoord){
// // 	vec3 albedo =texture(g_albedo_texture, pixelCoord).rgb; 
// // 	vec3 total_diffuse_reflectance_post_scatter = subsurface_scattering_total_diffuse_reflectance_post_scatter_from_albedo((postscatterEnabled > 0.0), albedo);
// // 	return total_diffuse_reflectance_post_scatter;
// // }
// float SSS_SUBSURFACE_MASK_SOURCE(vec2 pixelCoord){
// 	return texture(g_albedo_texture, pixelCoord).a;  //TODO maybe this should sample the skin texture
// }
// float SSS_PROJECTION_X_SOURCE(){
// 	return currProj[0][0];
// }
// float SSS_PROJECTION_Y_SOURCE(){
// 	return currProj[1][1];
// }
// vec2 uv_to_ndcxy(vec2 uv){
// 	return uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
// }
// float ndcz_to_viewpositionz(float ndcz, mat4 proj){
// 	return proj[3][2] / (ndcz - proj[2][2]);
// }
// vec3 ndc_to_view(vec3 ndc, mat4 proj){
// 	float view_position_z = ndcz_to_viewpositionz(ndc.z, proj);
// 	vec2 view_position_xy = ndc.xy * view_position_z / vec2(proj[0][0], proj[1][1]);
// 	return vec3(view_position_xy, view_position_z);
// }
// float SSS_VIEW_SPACE_POSITION_Z_SOURCE(vec2 pixelCoord){
// 	// NOTE: use SampleLevel rather than Sample
// 	// warning X3595: gradient instruction used in a loop with varying iteration; partial derivatives may have undefined value
// 	float depth = texture(depth_tex, pixelCoord).x; 
// 	float view_position_z = ndcz_to_viewpositionz(depth, currProj);
// 	return view_position_z;
// }
// vec3 SSS_VIEW_SPACE_POSITION_SOURCE(vec2 pixelCoord){
// 	float depth = texture(depth_tex, pixelCoord).x;  
// 	vec3 view_position = ndc_to_view(vec3(uv_to_ndcxy(pixelCoord), depth), currProj);
// 	return view_position;
// }
// vec2 SSS_PIXELS_PER_UV(){
// 	float outWidth;
// 	float outHeight;
// 	float outNumberOfLevels;
// 	// g_total_diffuse_reflectance_pre_scatter_multiply_form_factor_texture.GetDimensions(0, outWidth, outHeight, outNumberOfLevels);
// 	// return float2(outWidth, outHeight);
//     ivec2 tex_size= textureSize(g_total_diffuse_reflectance_pre_scatter_multiply_form_factor_texture, 0);
//     return vec2(tex_size.x, tex_size.y);
// }


// vec3 subsurface_scattering_disney_blur(const vec3 scattering_distance, const float world_scale, const int pixels_per_sample, const int sample_budget, const vec2 center_uv){
//     // const float dist_scale = SSS_SUBSURFACE_MASK_SOURCE(center_uv);
// 	// // Early Out
// 	// if (dist_scale < (1.0 / 255.0)){
// 	// 	vec3 total_diffuse_reflectance_pre_scatter_multiply_form_factor = SSS_TOTAL_DIFFUSE_REFLECTANCE_PRE_SCATTER_MULTIPLY_FORM_FACTOR_SOURCE(center_uv);

// 	// 	vec3 total_diffuse_reflectance_post_scatter = SSS_TOTAL_DIFFUSE_REFLECTANCE_POST_SCATTER_SOURCE(center_uv);
		
// 	// 	vec3 radiance = total_diffuse_reflectance_post_scatter * total_diffuse_reflectance_pre_scatter_multiply_form_factor;
// 	// 	return radiance;
// 	// }

//     // return vec3(1.0);

//     //try again
//     // float needs_sss_f=texture(metalness_and_roughness_and_sss_strength_tex, uv_in).z;
//     // bool needs_sss=needs_sss_f>0.5;
//     // if(needs_sss){
//     //     return vec3(1.0, 0.0, 0.0);
//     // }else{
//     //     return vec3(1.0, 1.0, 1.0);
//     // }

//     //attempt 2 
//     float needs_sss_f=texture(metalness_and_roughness_and_sss_strength_tex, uv_in).z;
//     float dist_scale=needs_sss_f;
//     // In UE4, the "dist_scale" is applied to the "scattering_distance" rather than the "uv_per_mm".  
// 	// This is equivalent since the "r" evaluated by the "diffusion_profile_sample_r" is proportional to the "scattering_distance".
// 	float meters_per_unit = world_scale;
//     float center_view_space_position_z = SSS_VIEW_SPACE_POSITION_Z_SOURCE(center_uv);
// 	float mms_per_unit = 1000.0 * meters_per_unit * (1.0 / dist_scale);
// 	vec2 uv_per_mm = 0.5 * vec2(SSS_PROJECTION_X_SOURCE(), SSS_PROJECTION_Y_SOURCE()) * (1.0 / center_view_space_position_z) * (1.0 / mms_per_unit);
// 	vec2 pixels_per_mm = SSS_PIXELS_PER_UV() * uv_per_mm;


//     const vec3 S = vec3(1.0, 1.0, 1.0) / scattering_distance.rgb;
// 	const float d = max(max(scattering_distance.r, scattering_distance.g), scattering_distance.b);


//     // Center Sample Reweighting
// 	//
// 	// The original burley algorithm involes monte car sampling. Given a random variable [0,1],
// 	// find the distance of that point from the center using the CDF, and then divide by PDF.
// 	// But it is somewhat inefficient because it is weighted heavily towards the center.
// 	//
// 	// Instead, we are going to split the [0,1] random variable range. First, we figure out the
// 	// radius (R) of the center sample in world space. Second, we are going to determine the random
// 	// variable (T) such that CDF(R) = T. Then we split the range into two segments.
// 	//
// 	// 1. The center sample, which include the random variable values from [0,T].
// 	// 2. All other samples, which include the random variable values from [T,1].
// 	//
// 	// With the center sample is scaled the weight T and the rest of the samples are weighted
// 	// by (1-T). There shouldn't be any bias, except for small errors due to precision.
// 	float center_sample_radius_in_mm = 0.5 * (1.0 / pixels_per_mm.x + 1.0 / pixels_per_mm.y);
// 	float center_sample_cdf = diffusion_profile_evaluate_cdf(d, center_sample_radius_in_mm);

// 	vec3 sum_numerator = vec3(0.0, 0.0, 0.0);
// 	vec3 sum_denominator = vec3(0.0, 0.0, 0.0);

// 	// Filter radius is, strictly speaking, infinite.
// 	// The magnitude of the function decays exponentially, but it is never truly zero.
// 	// To estimate the radius, we can use adapt the "three-sigma rule" by defining
// 	// the radius of the kernel by the value of the CDF which corresponds to 99.7%
// 	// of the energy of the filter.
// 	float filter_radius = diffusion_profile_sample_r(d, 0.997);
// 	float sample_count = int(float(PI) * (filter_radius * pixels_per_mm.x) * (filter_radius * pixels_per_mm.y) * (1.0 / float(max(pixels_per_sample, int(SSS_MIN_PIXELS_PER_SAMPLE)))));
// 	sample_count = min(sample_count, min(sample_budget, int(SSS_MAX_SAMPLE_BUDGET)));

//     for (int sample_index = 0; sample_index < int(SSS_MAX_SAMPLE_BUDGET) && sample_index < sample_count; ++sample_index){
// 		vec2 xi = hammersley_2d(sample_index, int(sample_count));

// 		// Center Sample Reweighting
// 		xi.x = mix(center_sample_cdf, 1.0, xi.x);

// 		// Sampling Diffusion Profile
// 		float r = diffusion_profile_sample_r(d, xi.x);
// 		float theta = 2.0 * float(PI) * xi.y;

// 		// Bilateral Filter
// 		vec2 sample_uv = center_uv + uv_per_mm * vec2(cos(theta), sin(theta)) * r;

// 		// The "sample_form_factor" may be zero even if the "sample_dist_scale" is NOT zero
// 		float sample_dist_scale = SSS_SUBSURFACE_MASK_SOURCE(sample_uv);

// 		if (sample_dist_scale >= (1.0 / 255.0)){
// 			// vec3 sample_total_diffuse_reflectance_pre_scatter_multiply_form_factor = SSS_TOTAL_DIFFUSE_REFLECTANCE_PRE_SCATTER_MULTIPLY_FORM_FACTOR_SOURCE(sample_uv);
//             vec3 sample_total_diffuse_reflectance_pre_scatter_multiply_form_factor=texture(composed_diffuse_gtex, sample_uv).rgb;

// 			float rcp_pdf = diffusion_profile_evaluate_rcp_pdf(d, r);

// 			// Bilateral Filter
// 			float sample_view_space_position_z = SSS_VIEW_SPACE_POSITION_Z_SOURCE(sample_uv);
// 			float relative_position_z_mm = mms_per_unit * (sample_view_space_position_z - center_view_space_position_z);
// 			float r_bilateral_weight = sqrt(r * r + relative_position_z_mm * relative_position_z_mm);

// 			vec3 pdf = diffusion_profile_evaluate_pdf(S, r_bilateral_weight);

// 			// normalized_diffusion_profile = pdf / r
// 			// (1.0 / float(N)) * total_diffuse_reflectance * (pdf / r) * form_factor * r * rcp_pdf 
// 			// = (1.0 / float(N)) * total_diffuse_reflectance * pdf * form_factor * rcp_pdf
// 			// = (1.0 / float(N)) * total_diffuse_reflectance_post_scatter * pdf * (total_diffuse_reflectance_pre_scatter * form_factor) * rcp_pdf
// 			// the 'total_diffuse_reflectance_post_scatter' will be calculated later while the 'total_diffuse_reflectance_pre_scatter * form_factor' is calculated here 
// 			vec3 sample_numerator = pdf * sample_total_diffuse_reflectance_pre_scatter_multiply_form_factor * rcp_pdf;

// 			// (1.0 / float(N)) * pdf * rcp_pdf
// 			vec3 sample_denominator = pdf * rcp_pdf;

// 			sum_numerator += sample_numerator;

// 			// assumed to be N since the pdf is normalized and the common divisor '1.0 / float(N)' is reduced
// 			sum_denominator += sample_denominator;
// 		}
// 	}

// 	// Center Sample Reweighting
// 	vec3 sum_total_diffuse_reflectance_pre_scatter_multiply_form_factor = sum_numerator / max(sum_denominator, vec3(FLT_MIN, FLT_MIN, FLT_MIN));
// 	// vec3 center_total_diffuse_reflectance_pre_scatter_multiply_form_factor = SSS_TOTAL_DIFFUSE_REFLECTANCE_PRE_SCATTER_MULTIPLY_FORM_FACTOR_SOURCE(center_uv);
// 	// vec3 total_diffuse_reflectance_pre_scatter_multiply_form_factor = lerp(sum_total_diffuse_reflectance_pre_scatter_multiply_form_factor, center_total_diffuse_reflectance_pre_scatter_multiply_form_factor, center_sample_cdf);

// 	// vec3 total_diffuse_reflectance_post_scatter = SSS_TOTAL_DIFFUSE_REFLECTANCE_POST_SCATTER_SOURCE(center_uv);
	
// 	// vec3 radiance = total_diffuse_reflectance_post_scatter * total_diffuse_reflectance_pre_scatter_multiply_form_factor;
// 	// return radiance;



//     // return vec3(1.0, 0.0, 0.0)*needs_sss_f;
//     // return vec3(vec2(pixels_per_mm),1.0)*needs_sss_f*0.001;
//     // return vec3(sample_count)*needs_sss_f*0.01;
//     return sum_total_diffuse_reflectance_pre_scatter_multiply_form_factor;
// }


// void main(){

//     vec3 scatteringDistance=vec3(0.8, 0.01, 0.01);
//     float padding_scatteringDistance=10.0;
//     float worldScale=1.0;
//     float postscatterEnabled=0.0;
//     int sampleBudget=256;
//     int pixelsPerSample=256;

//     vec3 color = subsurface_scattering_disney_blur(scatteringDistance, worldScale, pixelsPerSample, sampleBudget, uv_in);

    
//     blurred_output=vec4(vec3(color), 1.0);
// }




//old------------------------------------------------------------------------------------------------------------------------
// #version 330 core
// #extension GL_ARB_separate_shader_objects : require
// #extension GL_ARB_explicit_attrib_location : require

// //in
// layout(location=1) in vec2 uv_in;

// //out
// layout(location = 0) out vec4 blurred_output;

// uniform sampler2D img;

// // uniform bool horizontal;
// uniform float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
// // uniform int mip_map_lvl;

// void main(){

//     // blurred_output=vec4(1.0);

//     //https://learnopengl.com/Advanced-Lighting/Bloom
//     ivec2 tex_size= textureSize(img, 0);
//     vec2 tex_offset = 1.0 / tex_size; // gets size of single texel
//     vec4 result = textureLod(img, uv_in, 0) * weight[0]; // current fragment's contribution
//     for(int i = 1; i < 5; ++i){
//         result += texelFetch(img, ivec2(uv_in*tex_size + ivec2(i, 0)), 0) * weight[i];
//         result += texelFetch(img, ivec2(uv_in*tex_size - ivec2(i, 0)), 0) * weight[i];
//     }
    
//     blurred_output=result;
//     // if(dot(uv_in,uv_in)<1.0){
//         // blurred_output=vec4(1.0);
//     // }
//     // blurred_output=vec4(1.0);

// }



