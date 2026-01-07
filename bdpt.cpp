/**
 * @file bdpt.cpp
 * @brief Implementation of the Bidirectional Path Tracing (BDPT) integrator.
 *
 * This file contains the implementation of the BDPT integrator, which constructs
 * paths from both the camera and the light source and connects them to estimate
 * the radiance.
 *
 * Copyright (c) 2024. All rights reserved.
 * This code is modified from the Nori renderer.
 */

#include <algorithm>
#include <vector>
#include <atomic>
#include <string>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>

#include <nori/bsdf.h>
#include <nori/camera.h>
#include <nori/emitter.h>
#include <nori/integrator.h>
#include <nori/mesh.h>
#include <nori/sampler.h>
#include <nori/scene.h>

NORI_NAMESPACE_BEGIN

enum VertexType { VERTEX_DIFFUSE_BDPT, VERTEX_SPECULAR_BDPT };

struct PathVertex {
    bool is_startpoint;
    VertexType type;

    // Location
    Intersection its;
    Vector3f w_e2l; // direction from eye to light in local frame
    Vector3f w_l2e; // direction from light to eye in local frame

    // Increment probability of current vertex from eye subpath direction (camera)
    double p_eye;
    double p_eye_geo;

    // Increment probability of current vertex from light subpath direction
    // (emitter)
    double p_light;
    double p_light_geo;

    // Throughput from eye subpath direction.
    Color3f throughput_eye;

    // Throughput from light subpath direction.
    Color3f throughput_light;

    std::string toString() const {
        return "PathVertex{ is_startpoint=" + std::to_string(is_startpoint) +
            ", type=" + std::to_string(type) +
            ", its=" + its.p.toString() +
            ", w_e2l=" + w_e2l.toString() +
            ", w_l2e=" + w_l2e.toString() +
            ", p_eye=" + std::to_string(p_eye) +
            ", p_eye_geo=" + std::to_string(p_eye_geo) +
             ", p_light=" + std::to_string(p_light) +
            ", p_light_geo=" + std::to_string(p_light_geo) +
            ", throughput_eye=" + throughput_eye.toString() +
            ", throughput_light=" + throughput_light.toString() +
            "}";
    }

    // For endpoint of eye subpath, update p_light, p_light_geo, throughput_eye,
    // throughput_light, and w_e2l
    void update_eye_direction_next_vertex(const PathVertex& next_vertex,
        const double RR_pdf, bool isSample=true) {
        auto distance = (next_vertex.its.p - its.p).norm();
        auto direction = (next_vertex.its.p - its.p) / distance;
        w_e2l = its.shFrame.toLocal(direction);

        if (next_vertex.type == VERTEX_SPECULAR_BDPT) {
            p_light = 1.0;
            p_light_geo = 1.0;
        } else {
            if (next_vertex.its.mesh->isEmitter())
            {
                auto emitter = next_vertex.its.mesh->getEmitter();
                EmitterQueryRecord emitter_query_record(next_vertex.its.p,
                    next_vertex.its.shFrame);
                double direction_pdf = (double)emitter->directionPdf(emitter_query_record);
                double G = (double)Frame::cosTheta(w_e2l) / (distance * distance);
                G = abs(G);
                
                p_light = direction_pdf;
                p_light_geo = G * p_light;
            } else {
                // P_light consists of three part: bsdf pdf, g term, and RR
                BSDFQueryRecord bsdf_query_record(
                    next_vertex.w_e2l, next_vertex.its.shFrame.toLocal(-direction),
                    ESolidAngle);
                double bsdf_pdf = (double)next_vertex.its.mesh->getBSDF()->pdf(bsdf_query_record);
                double G = (double)Frame::cosTheta(w_e2l) / (distance * distance);

                if (type == VERTEX_SPECULAR_BDPT) // Handle refraction
                    G = abs(G);

                p_light = bsdf_pdf * RR_pdf;
                p_light_geo = p_light * G;
            }
        }

        // Update the throughput eye
        if (is_startpoint)
        {
            throughput_eye = Color3f(1.0f); // No weight here
            throughput_light = Color3f(1.0f); // No weight function here
            if (!isSample)
            {
                throughput_light *= 1.0f / (distance * distance);
            }
        } else {
            if (type == VERTEX_SPECULAR_BDPT)
            {
                throughput_eye = Color3f(1.0f);
                throughput_light = Color3f(1.0f);  // Can be fresnel later, since we omit the specular point, so it is OK.
            } else {
                BSDFQueryRecord bRecThroughput(w_e2l, w_l2e, ESolidAngle);
                throughput_eye =
                    its.mesh->getBSDF()->eval(bRecThroughput) * its.shFrame.cosTheta(w_l2e);

                // Update the throughput light
                BSDFQueryRecord bRecThroughputLight(w_l2e, w_e2l, ESolidAngle);
                throughput_light = its.mesh->getBSDF()->eval(bRecThroughputLight) *
                    its.shFrame.cosTheta(w_e2l);

                if (!isSample)
                {
                    throughput_light *= Frame::cosTheta(next_vertex.w_l2e) / (distance * distance);
                }
            }
        }
    }

    // For endpoint of light subpath, update p_eye, p_eye_geo, throughput_eye,
    // throughput_light, and w_l2e
    void update_light_direction_next_vertex(const PathVertex& next_vertex,
        const double RR_pdf, const Scene* scene, bool isSample=true) {
        auto distance = (next_vertex.its.p - its.p).norm();
        if (distance <= 1e-10f) {
            throw std::runtime_error("distance <= 1e-10f");
        }
        auto direction = (next_vertex.its.p - its.p) / distance;
        w_l2e = its.shFrame.toLocal(direction);

        if (next_vertex.type == VERTEX_SPECULAR_BDPT) {
            p_eye = 1.0;
            p_eye_geo = 1.0;
        } else {
            // P_eye consists of three part: bsdf pdf, g term, and RR
            if (next_vertex.is_startpoint) {
                // The sampling strategy has been changed to the camera samplig
                // strategy, so does the sample pdf.
                auto camera = scene->getCamera();
                auto camera_view_direction = camera->getCameraViewDirection();
                double cosTheta = (double)camera_view_direction.normalized().dot(-direction);
                double sample_pdf = 0.25 * 1.0 / (cosTheta * cosTheta * cosTheta);
                double G = (double)Frame::cosTheta(w_l2e) / (distance * distance);

                if (!its.mesh->getBSDF()->isDiffuse())
                    G = abs(G);

                p_eye = sample_pdf;
                p_eye_geo = sample_pdf * G;
            } else {
                BSDFQueryRecord bsdf_query_record(
                next_vertex.w_l2e, next_vertex.its.shFrame.toLocal(-direction), 
                ESolidAngle);
                double bsdf_pdf = (double)next_vertex.its.mesh->getBSDF()->pdf(bsdf_query_record);
                double G = (double)Frame::cosTheta(w_l2e) / (distance * distance);

                if (type == VERTEX_SPECULAR_BDPT) // Handle refraction
                    G = abs(G);

                p_eye = bsdf_pdf * RR_pdf;
                p_eye_geo = p_eye * G;
            }
        }

        // Update the throughput light
        if (its.mesh->isEmitter()) {
            EmitterQueryRecord eRec(its.p, its.shFrame);
            throughput_light = its.mesh->getEmitter()->eval(eRec) * Frame::cosTheta(w_l2e);
            throughput_eye = Color3f(1.0f);
        } else {
            if (type == VERTEX_SPECULAR_BDPT)
            {
                throughput_eye = Color3f(1.0f);
                throughput_light = Color3f(1.0f);  // Can be fresnel later, since we omit the specular point, so it is OK.
            } else {
                // Update the throughput eye
                BSDFQueryRecord bRecThroughput(w_e2l, w_l2e, ESolidAngle);
                throughput_eye = its.mesh->getBSDF()->eval(bRecThroughput) *
                    its.shFrame.cosTheta(w_e2l);

                BSDFQueryRecord bRecThroughputLight(w_l2e, w_e2l, ESolidAngle);
                throughput_light = its.mesh->getBSDF()->eval(bRecThroughputLight) * its.shFrame.cosTheta(w_l2e); // The cosine term is left by the cancel out of the g term at both numerator and denominator, thus the cosine term is dependent to the probability direction

                if (!isSample)
                {
                    throughput_eye *= Frame::cosTheta(next_vertex.w_l2e) / (distance * distance);
                }
            }
        }
    }
};

// Create path enum
enum PathType { PATH_EYE_SUBPATH, PATH_LIGHT_SUBPATH, PATH_MERGED };

/*
    The integration of path and subpath.
*/
struct Path {
    std::vector<PathVertex> vertices;
    int num_eye_vertices = 0;
    int num_light_vertices = 0;
    PathType type;
    double g_term = 1.0;  // The additional g term from the merging.

    Path(PathType type) : type(type) {}

    size_t length() const { return vertices.size(); }

    void generate_child_paths(std::vector<Path>& child_paths, bool excludeT1 = true) {
        child_paths.clear();
        child_paths.push_back(*this);

        if (type == PATH_EYE_SUBPATH) {
            // The t=0 are excluded.
            auto len = length();
            for (size_t i = excludeT1 ? 1 : 0; i < len - 1; i++) {
                // skip if the endpoint is specular
                if (vertices[len - i - 1].type == VERTEX_SPECULAR_BDPT)
                {
                    continue;
                }

                // create new path
                Path child_path(type);

                auto child_vertices_len = len - i;
                child_path.vertices.resize(child_vertices_len);
                std::copy(vertices.begin(), vertices.begin() + child_vertices_len,
                    child_path.vertices.begin());

                
                child_path.num_eye_vertices = num_eye_vertices - i > 0 ? num_eye_vertices - i : 0;
                child_path.num_light_vertices = num_light_vertices - i > 0 ? num_light_vertices - i : 0;

                child_paths.push_back(child_path);
            }
        } else if (type == PATH_LIGHT_SUBPATH) {
            // The s=0 is excluded.
            auto len = length();
            for (size_t i = 1; i < len; i++) {
                // skip if the endpoint is specular
                if (vertices[len - i - 1].type == VERTEX_SPECULAR_BDPT)
                {
                    continue;
                }

                // create new path
                Path child_path(type);

                auto child_vertices_len = len - i;
                child_path.vertices.resize(child_vertices_len);
                std::copy(vertices.begin(), vertices.begin() + child_vertices_len,
                    child_path.vertices.begin());

                child_path.num_eye_vertices = num_eye_vertices - i > 0 ? num_eye_vertices - i : 0;
                child_path.num_light_vertices = num_light_vertices - i > 0 ? num_light_vertices - i : 0;

                child_paths.push_back(child_path);
                
            }
        }
    }

    void append_path_vertex(Intersection& its, const double RR_pdf,
        const Scene* scene, const Emitter* emitter = nullptr,
        const double startpoint_pdf = 1.0) {
        // Convert the intersection into a valid path vertex.
        if (type == PATH_EYE_SUBPATH) {
            num_eye_vertices++;
        } else if (type == PATH_LIGHT_SUBPATH) {
            num_light_vertices++;
        } else {
            throw std::runtime_error("Invalid path type [" + std::to_string(type) +
                "]: Cannot append a vertex to a merged path!");
        }

        // is endpoint
        bool is_startpoint = length() == 0;
        VertexType vertex_type = (is_startpoint || its.mesh->getBSDF()->isDiffuse())
            ? VERTEX_DIFFUSE_BDPT
            : VERTEX_SPECULAR_BDPT;

        // w_l2e, p_eye, p_eye_geo ( for eye subpath )
        // w_e2l, p_light, p_light_geo ( for light subpath )
        Vector3f w_l2e(0.0f, 0.0f, 1.0f), w_e2l(0.0f, 0.0f, 1.0f);
        double p_eye = 1.0, p_eye_geo = 1.0, p_light = 1.0, p_light_geo = 1.0;
        if (is_startpoint) {
            if (type == PATH_EYE_SUBPATH) {
                // The w_l2e is useless
                w_l2e = Vector3f(0.0f, 0.0f, 1.0f);
                p_eye = startpoint_pdf;
                p_eye_geo = startpoint_pdf;
            } else if (type == PATH_LIGHT_SUBPATH) {
                // The w_e2l is useless
                w_e2l = Vector3f(0.0f, 0.0f, 1.0f);
                p_light = startpoint_pdf;
                p_light_geo = startpoint_pdf;
            } else {
                throw std::runtime_error("Invalid path type [" + std::to_string(type) +
                    "]: Cannot append a vertex to a merged path!");
            }
        } else {
            // Get the current endpoint
            PathVertex& endpoint = vertices.back();
            if (type == PATH_EYE_SUBPATH) {
                auto direction_vector = its.p - endpoint.its.p;
                auto distance = direction_vector.norm();
                auto direction = direction_vector / distance;
                w_l2e = its.shFrame.toLocal(-direction);

                if (!endpoint.is_startpoint && endpoint.type == VERTEX_SPECULAR_BDPT) {
                    // transport the p_eye directly
                    p_eye = 1.0;
                    p_eye_geo = 1.0;
                } else {
                    if (endpoint.is_startpoint) {
                        // The sampling strategy has been changed to the camera samplig
                        // strategy, so does the sample pdf.
                        auto camera = scene->getCamera();
                        auto camera_view_direction = camera->getCameraViewDirection();
                        double cosTheta = (double)camera_view_direction.normalized().dot(direction);
                        double sample_pdf = 0.25 * 1.0 / (cosTheta * cosTheta * cosTheta);
                        double G = (double)Frame::cosTheta(w_l2e) / (distance * distance);

                        if (!its.mesh->getBSDF()->isDiffuse())
                            G = abs(G);

                        p_eye = sample_pdf;
                        p_eye_geo = sample_pdf * G;
                    } else {
                        BSDFQueryRecord bsdf_query_record(
                            endpoint.w_l2e, endpoint.its.shFrame.toLocal(direction),
                            ESolidAngle);
                        double bsdf_pdf =
                            (double)endpoint.its.mesh->getBSDF()->pdf(bsdf_query_record);
                        double G = (double)Frame::cosTheta(w_l2e) / (distance * distance);

                        if (!its.mesh->getBSDF()->isDiffuse())
                            G = abs(G);

                        p_eye = bsdf_pdf;
                        p_eye_geo = p_eye * G;
                    }
                }

            } else if (type == PATH_LIGHT_SUBPATH) {
                auto direction_vector = its.p - endpoint.its.p;
                auto distance = direction_vector.norm();
                auto direction = direction_vector / distance;
                w_e2l = its.shFrame.toLocal(-direction);

                if (!endpoint.is_startpoint && endpoint.type == VERTEX_SPECULAR_BDPT) {
                    // transport the p_light directly
                    p_light = 1.0;
                    p_light_geo = 1.0;
                } else {
                    if (endpoint.is_startpoint) {
                        if (emitter == nullptr) {
                            throw std::runtime_error(
                                "BDPTIntegrator::preprocess(): Emitter is nullptr! You "
                                "need "
                                "to "
                                "provide the emitter during the preprocess light subpath "
                                "creation.");
                        }
                        // sample on emitter
                        EmitterQueryRecord emitter_query_record(endpoint.its.p,
                            endpoint.its.shFrame);
                        double direction_pdf = (double)emitter->directionPdf(emitter_query_record);
                        double G = (double)Frame::cosTheta(w_e2l) / (distance * distance);
                        if (!its.mesh->getBSDF()->isDiffuse())
                            G = abs(G);
                        p_light = direction_pdf;
                        p_light_geo = G * p_light;
                    } else {
                        BSDFQueryRecord bsdf_query_record(
                            endpoint.w_e2l, endpoint.its.shFrame.toLocal(direction),
                            ESolidAngle);

                        double bsdf_pdf =
                            (double)endpoint.its.mesh->getBSDF()->pdf(bsdf_query_record);
                        double G = (double)Frame::cosTheta(w_e2l) / (distance * distance);
                        if (!its.mesh->getBSDF()->isDiffuse())
                            G = abs(G);
                        p_light = bsdf_pdf * RR_pdf;
                        p_light_geo = p_light * G;

                    }
                }
            } else {
                throw std::runtime_error("Invalid path type [" + std::to_string(type) +
                    ": Cannot append a vertex to a merged path!");
            }
        }

        // Create new path vertex
        PathVertex new_vertex;
        new_vertex.is_startpoint = is_startpoint;
        new_vertex.type = vertex_type;
        new_vertex.its = its;
        new_vertex.w_e2l = w_e2l;
        new_vertex.w_l2e = w_l2e;
        new_vertex.p_eye = p_eye;
        new_vertex.p_eye_geo = p_eye_geo;
        new_vertex.p_light = p_light;
        new_vertex.p_light_geo = p_light_geo;
        new_vertex.throughput_eye = Color3f(1.0f);
        new_vertex.throughput_light = Color3f(1.0f);

        if (is_startpoint) {
            vertices.push_back(new_vertex);
        } else {
            PathVertex& prev_endpoint = vertices.back();
            if (type == PATH_EYE_SUBPATH) {
                prev_endpoint.update_eye_direction_next_vertex(new_vertex, RR_pdf);
            } else if (type == PATH_LIGHT_SUBPATH) {
                prev_endpoint.update_light_direction_next_vertex(new_vertex, RR_pdf, scene);
            } else {
                throw std::runtime_error("Invalid path type [" + std::to_string(type) +
                    ": Cannot append a vertex to a merged path!");
            }
            vertices.push_back(new_vertex);
        }
    }

    void postprocess_subpath_gen()
    {
        // Do it before merge
        if (type == PATH_MERGED) {
            throw std::runtime_error("Invalid path type [" + std::to_string(type) +
                ": Cannot do postprocess!");
        }
        
        // 1. Specular chain cut
        // If the endpoint is specular, drop it, until it is not a specular
        while (vertices.back().type == VERTEX_SPECULAR_BDPT) {
            vertices.pop_back();
            if (type == PATH_EYE_SUBPATH) num_eye_vertices--;
            else num_light_vertices--;
        }

    }

    void mark_as_merged() {
        // Only Eye subpath can be marked as merged.
        if (type != PATH_EYE_SUBPATH)
        {
            throw std::runtime_error("Invalid path type [" + std::to_string(type) +
                ": Cannot mark as merged!");
        }
        // The endpoint need to hit on the emitter
        if (!vertices.back().its.mesh->isEmitter())
        {
            throw std::runtime_error("Invalid path endpoint [" + std::to_string(type) +
                ": Cannot mark as merged!");
        }

        // Fill up the info of the endpoint, since it miss the updation
        // including p_light, p_light_geo, throughput_eye,
        // throughput_light, and w_e2l
        auto& endpoint = vertices.back();
        // p_light and p_light_geo
        endpoint.p_light = (double)endpoint.its.mesh->pdf(endpoint.its);
        endpoint.p_light_geo = endpoint.p_light;
        // throughput_eye
        endpoint.throughput_eye = Color3f(1.0f);
        // throughput_light is the emitter evaluation
        EmitterQueryRecord eRec(endpoint.its.p, endpoint.its.shFrame);
        endpoint.throughput_light = endpoint.its.mesh->getEmitter()->eval(eRec);
        
        // w_e2l (useless)
        endpoint.w_e2l = Vector3f(0.0f, 0.0f, 1.0f);

        type = PATH_MERGED; } // For the s=0 technique.

    void merge(Path& eye_subpath, Path& light_subpath, Path& mergedPath,
        double RR_pdf, const Scene* scene) {
        auto len_eye_subpath = eye_subpath.length();
        auto len_light_subpath = light_subpath.length();

        PathVertex eye_endpoint = eye_subpath.vertices[len_eye_subpath - 1]; // Need to copy!
        PathVertex light_endpoint = light_subpath.vertices[len_light_subpath - 1]; // Need to copy!
        // Update the endpoint vertex of both the two subpaths
        light_endpoint.w_l2e = light_endpoint.its.shFrame.toLocal((eye_endpoint.its.p - light_endpoint.its.p).normalized());
        eye_endpoint.update_eye_direction_next_vertex(light_endpoint, RR_pdf, false);
        light_endpoint.update_light_direction_next_vertex(eye_endpoint, RR_pdf, scene, false);

        // Merge the two subpaths into one path
        mergedPath.vertices.resize(len_eye_subpath + len_light_subpath);
        std::copy(eye_subpath.vertices.begin(), eye_subpath.vertices.end(),
            mergedPath.vertices.begin());
        // reverse the light subpath (should not use the reverse, just copy in the reverse
        for (int i = 0; i < len_light_subpath; i++) {
            PathVertex temp = light_subpath.vertices[len_light_subpath - 1 - i];
            mergedPath.vertices[len_eye_subpath + i] = temp;
        }

        // Replace with the new endpoint
        mergedPath.vertices[len_eye_subpath - 1] = eye_endpoint;
        mergedPath.vertices[len_eye_subpath] = light_endpoint;

        num_eye_vertices = len_eye_subpath;
        num_light_vertices = len_light_subpath;
        type = PATH_MERGED;

        double g = 1.0;
        g_term = g;
    }

    Color3f compute_contribution() const {
        // We need to compute:
        // 1. accumulated throughput
        // 2. accumulated pdf
        // 3. mis weight

        // Accumulate the throughput ( direction is from light to eye )
        Color3f throughput(1.0f);
        for (const PathVertex& vertex : vertices) {
            throughput *= vertex.throughput_light;
        }
        throughput *= g_term;

        // Accumulate the pdf
        double pdf = 1.0;
        double pdf_g = 1.0;
        double pdf_g_inv = 1.0;
        // Compute eye subpath pdf
        for (int i = 0; i < num_eye_vertices; i++) {
            pdf *= vertices[i].p_eye;
            pdf_g *= vertices[i].p_eye_geo;
            pdf_g_inv *= 1.0 / (vertices[i].p_eye_geo + 1e-8);
        }
        // Compute light subpath pdf
        for (int i = num_eye_vertices; i < num_eye_vertices + num_light_vertices;
            i++) {
            pdf *= vertices[i].p_light;
            pdf_g *= vertices[i].p_light_geo;
            pdf_g_inv *= 1.0 / (vertices[i].p_light_geo + 1e-8);
        }

        // Compute the mis weight beta = 2 balance heuristic
        double total_pdf = 0.0;
        for (int eye_endpoint_idx = 0;
            eye_endpoint_idx < num_eye_vertices + num_light_vertices;
            eye_endpoint_idx++) {
            if (vertices[eye_endpoint_idx].type == VERTEX_SPECULAR_BDPT || (eye_endpoint_idx < num_eye_vertices + num_light_vertices - 1 && vertices[eye_endpoint_idx + 1].type == VERTEX_SPECULAR_BDPT))
                continue;

            if (eye_endpoint_idx == 0 &&  vertices[eye_endpoint_idx + 1].type == VERTEX_SPECULAR_BDPT && num_eye_vertices != 1) // We don't handle the event that camera frist hit point is one the specular obj.
                continue;

            double current_pdf = 1.0;
            double current_pdf_inv = 1.0;
            for (int i = 0; i <= eye_endpoint_idx; i++) {
                current_pdf *= vertices[i].p_eye_geo;
                current_pdf_inv *= 1.0 / (vertices[i].p_eye_geo + 1e-8);
            }
            for (int i = eye_endpoint_idx + 1;
                i < num_eye_vertices + num_light_vertices; i++) {
                current_pdf *= vertices[i].p_light_geo;
                current_pdf_inv *= 1.0 / (vertices[i].p_light_geo + 1e-8);
            }

            // This is a solution to avoid precision lost error.
            double curPdfDIVpdf;
            if (pdf_g < 1e-6)
            {
                // we use inv
                curPdfDIVpdf = pdf_g_inv / (current_pdf_inv + 1e-8);
            }
            else
            {
                curPdfDIVpdf = current_pdf / ( pdf_g + 1e-8);
            }

            total_pdf += curPdfDIVpdf;
        }

        
        double mis_weight = 1.0 / (total_pdf + 1e-8);
        mis_weight = std::min(mis_weight, 1.0);

        Color3f contribution = Color3f(0.0f) + (float)mis_weight * throughput / (float)(pdf + 1e-8);

        return contribution;
    }

    std::string toString() const {
        std::string ret = "Path{\n";
        // length
        ret += "  length = " + std::to_string(length()) + "\n";
        for (const auto& vertex : vertices)
            ret += "  " + vertex.toString() + "\n";
        ret += "}";
        return ret;
    }
};

class BDPTIntegrator : public Integrator {
public:
    BDPTIntegrator(const PropertyList& props) {
        m_rr = props.getFloat("rr", 0.95f); // Currently the RR is fixed.
        m_numLightImageRay = props.getInteger("numLightImageRay", 1e5);
    }

    void preprocess(const Scene* scene, ImageBlock& lightImage) override {
        // Calculate the light image, which is necessary for caustics

        const Camera* camera = scene->getCamera();
        auto outputSize = camera->getOutputSize();
        float scaling_factor = (float)(outputSize.x() * outputSize.y()) / (float)m_numLightImageRay;
        // Sampler* sampler = scene->getSampler_nonconst();

        /* Create a block generator (i.e. a work scheduler) */
        printf("Start light image sampling\n");
        
        static std::atomic<int> seed(0);

        tbb::blocked_range<int> range(0, m_numLightImageRay);

        // Define thread-local state
        struct ThreadState {
            std::unique_ptr<Sampler> sampler;
            std::unique_ptr<ImageBlock> block;
        };

        tbb::enumerable_thread_specific<ThreadState> threadStates([&]() {
            ThreadState state;
            state.sampler = scene->getSampler()->clone();
            state.block = std::unique_ptr<ImageBlock>(new ImageBlock(lightImage.getSize(), camera->getReconstructionFilter()));
            state.block->clear();
            
            // Seed the sampler
            ImageBlock dummyBlock(Vector2i(1), nullptr);
            dummyBlock.setOffset(Point2i(seed++, 0));
            state.sampler->prepare(dummyBlock);
            
            return state;
        });

        auto map = [&](const tbb::blocked_range<int> &range) {
            ThreadState& state = threadStates.local();
            Sampler *sampler = state.sampler.get();
            ImageBlock *block = state.block.get();

            for (int i=range.begin(); i<range.end(); ++i) {
                // Generate light subpath
                Path light_subpath(PATH_LIGHT_SUBPATH);
                trace_light_subpath(light_subpath, scene, sampler);
                while (light_subpath.vertices.back().its.mesh->isEmitter() && light_subpath.length() != 1) {
                    light_subpath = Path(PATH_LIGHT_SUBPATH);
                    trace_light_subpath(light_subpath, scene, sampler);
                }
                light_subpath.postprocess_subpath_gen();

                // Only camera point 
                Path eye_subpath(PATH_EYE_SUBPATH);
                Intersection start_its;
                start_its.p = camera->getCameraPosition();
                start_its.shFrame = Frame(camera->getCameraViewDirection().normalized());
                eye_subpath.append_path_vertex(start_its, m_rr, scene, nullptr, 1.0f);

                std::vector<Path> merged_path;
                generate_paths(eye_subpath, light_subpath, merged_path, scene, false);


                if (merged_path.size() > 0)
                {
                    for (Path& path : merged_path)
                    {
                        // Get the direction
                        Vector3f direction = (path.vertices[1].its.p - path.vertices[0].its.p).normalized();
                        Color3f contribution = path.compute_contribution() * scaling_factor;

                        // Get sample pos
                        Point2f samplePos;
                        bool isHit = camera->getPixelCoordinate(direction, samplePos);
                        if (isHit)
                        {
                            // store the contribution into light image.
                            block->put(samplePos, contribution);
                        }
                    }
                }
            }
        };

        tbb::parallel_for(range, map);

        // Merge all blocks
        for (const auto& state : threadStates) {
            lightImage.put(*state.block);
        }
        
        printf("Light image finished\n");
    }

    void trace_light_subpath(Path& path, const Scene* scene,
        Sampler* sampler) const {

        // Sample an emitter from scene
        float emitterSamplePDF = 0.0f;
        auto emitter = scene->sampleEmitter(sampler->next1D(), emitterSamplePDF);

        // Sample a point on the emitter
        auto nested_mesh = emitter->getNestedMesh();
        if (nested_mesh == nullptr) {
            // This is a point emitter or directional emitter
            // Currently not available.
            throw std::runtime_error(
                "BDPTIntegrator::preprocess(): Point emitter or directional "
                "emitter is not available currently!");
        }
        Intersection start_its;
        auto surfSamplePDF = nested_mesh->sample(start_its, sampler);
        path.append_path_vertex(start_its, m_rr, scene, emitter,
            emitterSamplePDF * surfSamplePDF);

        // sample a ray and start the tracing.
        EmitterQueryRecord eRec(start_its.p, start_its.shFrame);
        emitter->sampleDirection(eRec, sampler);
        Ray3f ray(start_its.p, eRec.sh_frame.toWorld(eRec.d));
        ray.update();

        // Trace the light subpath
        trace_subpath(path, scene, sampler, ray, emitter);
    }

    void trace_eye_subpath(Path& path, const Scene* scene, Sampler* sampler,
        const Ray3f& ray) const {
        Intersection start_its;
        start_its.p = ray.o;
        start_its.shFrame = Frame(scene->getCamera()->getCameraViewDirection().normalized());

        path.append_path_vertex(start_its, m_rr, scene, nullptr, 1.0f);

        trace_subpath(path, scene, sampler, ray);
    }

    void trace_subpath(Path& path, const Scene* scene, Sampler* sampler,
        const Ray3f& _ray, Emitter *emitter=nullptr) const {
        auto shouldTerminate = false;

        Intersection its;
        Ray3f ray(_ray);
        while (!shouldTerminate) {
            if (!scene->rayIntersect(ray, its))
                break;

            path.append_path_vertex(its, m_rr, scene, emitter, 0.0f);

            // if emitter, early stop
            if (its.mesh->isEmitter()) {
                shouldTerminate = true;
            }

            // Sample the next direction by BSDF sampling
            BSDFQueryRecord bRec(its.shFrame.toLocal(-ray.d));
            auto bsdf = its.mesh->getBSDF();
            bsdf->sample(bRec, sampler->next2D());

            if (its.mesh->getBSDF()->isDiffuse() && sampler->next1D() > m_rr) {
                shouldTerminate = true;
            }

            ray.o = its.p;
            ray.d = its.shFrame.toWorld(bRec.wo);
            ray.mint = 1e-3f;
            ray.update();
        }
    }

    void generate_paths(Path& eye_subpath, Path& light_subpath,
        std::vector<Path>& merged_path_list,
        const Scene* scene, bool excludeT1 = true) const {
        // Generate the child path for both to subpath
        std::vector<Path> eye_subpath_list;
        std::vector<Path> light_subpath_list;

        eye_subpath.generate_child_paths(eye_subpath_list, excludeT1);
        light_subpath.generate_child_paths(light_subpath_list);

        for (auto& eye_subpath : eye_subpath_list) {
            for (auto& light_subpath : light_subpath_list) {
                // Test the visibility between the two endpoints
                Vector3f shadowRayDir = light_subpath.vertices.back().its.p -
                    eye_subpath.vertices.back().its.p;
                float distance = shadowRayDir.norm();

                shadowRayDir.normalize();
                Ray3f shadowRay(eye_subpath.vertices.back().its.p, shadowRayDir);
                shadowRay.mint = 1e-4f;
                shadowRay.maxt = distance - 1e-4f;
                shadowRay.update();
                if (scene->rayIntersect(shadowRay)) {
                    continue;
                }

                // additional test to avoid negative costheta
                if (shadowRayDir.dot(light_subpath.vertices.back().its.shFrame.n) >= 0.0f
                    ||
                    shadowRayDir.dot(eye_subpath.vertices.back().its.shFrame.n) <= 0.0f)
                {
                    continue;
                }
                
                Path merged_path(PATH_MERGED);
                merged_path.merge(eye_subpath, light_subpath, merged_path, m_rr, scene);
                merged_path_list.push_back(merged_path);
                
            };
        }
    }

    Color3f Li(const Scene* scene, Sampler* sampler,
        const Ray3f& ray, std::vector<Color3f>& otherContribution) const override {
        std::vector<Path>
            merged_paths; // All the path that need to compute the contribution

        Path light_subpath(PATH_LIGHT_SUBPATH);
        // Sample a light subpath
        trace_light_subpath(light_subpath, scene, sampler);
        // If the light subpath is ended at emitter, regenerate the path
        while (light_subpath.vertices.back().its.mesh->isEmitter() && light_subpath.length() != 1) {
            light_subpath = Path(PATH_LIGHT_SUBPATH);
            trace_light_subpath(light_subpath, scene, sampler);
        }
        light_subpath.postprocess_subpath_gen();

        // Sample an eye subpath
        Path eye_subpath(PATH_EYE_SUBPATH);
        trace_eye_subpath(eye_subpath, scene, sampler, ray);
        eye_subpath.postprocess_subpath_gen();

        if (eye_subpath.length() == 1) {
            return Color3f(0.0f); // early stop
        }

        // If the eye subpath is ended at emitter, mark it as a merged path
        if (eye_subpath.vertices.back().its.mesh &&
            eye_subpath.vertices.back().its.mesh->isEmitter()) {
            eye_subpath.mark_as_merged();
            merged_paths.push_back(eye_subpath);
        }
        else{
            // Merge the two subpaths
            generate_paths(eye_subpath, light_subpath, merged_paths, scene);
        }

        Color3f contribution(0.0f);
        if (merged_paths.size() > 0) {
            contribution = merged_paths[0].compute_contribution();
            
            for (int i = 1; i < merged_paths.size(); i++)
                otherContribution.push_back(merged_paths[i].compute_contribution());
        }

        return contribution;
    }

    std::string toString() const override { return "BDPTIntegrator["
        "m_rr=" + std::to_string(m_rr) +
        "]"; }

private:
    float m_rr;
    int m_numLightImageRay;
};

NORI_REGISTER_CLASS(BDPTIntegrator, "bdpt");
NORI_NAMESPACE_END
