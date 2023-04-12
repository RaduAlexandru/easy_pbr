#version 330 core
#extension GL_ARB_separate_shader_objects : require
#extension GL_ARB_explicit_attrib_location : require

//following mostly https://github.com/DoerriesT/Separable-Subsurface-Scattering-Demo/blob/master/SubsurfaceScattering/resources/shaders/sssBlur_comp.comp 

//in
layout(location=1) in vec2 uv_in;

//out
layout(location = 0) out vec4 blurred_output;

uniform sampler2D composed_diffuse_gtex;
//TODO need
uniform sampler2D depth_tex;
uniform sampler2D metalness_and_roughness_and_sss_strength_tex;

// uniform bool horizontal;
uniform float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

//TODO to be filled
uniform mat4 currProj;
uniform float projection_a; //for calculating position from depth according to the formula at the bottom of article https://mynameismjp.wordpress.com/2010/09/05/position-from-depth-3/
uniform float projection_b;
uniform float fov_x;
uniform bool is_ortho;
uniform bool horizontal;
uniform float sss_width;


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
    vec2 dir;
    if (horizontal){
        dir=vec2(1.0, 0.0);   
    }else{
        dir=vec2(0.0, 1.0);  
    }
    float sss_width_world=sss_width;
    // float maxDepthDiff = 0.1;


 

    float depth_val = texture(depth_tex, uv_in).x;
    float linear_depth_val=linear_depth(depth_val);

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
	
    vec2 texCoord=uv_in;
	
	


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

    
    blurred_output=vec4(colorBlurred);
}




















