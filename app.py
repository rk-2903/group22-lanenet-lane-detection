import sys
import os

# Add the lanenet-lane-detection directory to the Python path
lanenet_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'lanenet-lane-detection'))
sys.path.append(lanenet_dir)

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import tensorflow as tf

# LaneNet-specific imports
try:
    from lanenet_model import lanenet
    from lanenet_model import lanenet_postprocess
    from local_utils.config_utils import parse_config_utils
    from local_utils.log_util import init_logger
except ImportError as e:
    st.error(f"Failed to import LaneNet modules: {str(e)}")
    st.write("Ensure the 'lanenet-lane-detection' directory is in the correct location and contains the required modules.")
    st.write("Also ensure that lanenet_model/ and local_utils/ have an __init__.py file.")
    st.stop()


# Configuration for LaneNet
CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

# Ensure TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()

def detect_lanes(input_data, input_type="image", confidence_threshold=0.7, weights_path=None):
    if weights_path is None:
        raise ValueError("Weights path must be provided for LaneNet model inference.")

    # Initialize TensorFlow session and model
    input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')
    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # Set session configuration
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    sess = tf.compat.v1.Session(config=sess_config)

    # Load model weights
    with tf.compat.v1.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess=sess, save_path=weights_path)

    if input_type == "image":
        # Preprocess the image
        image = input_data.copy()
        image_vis = image.copy()  # Keep a copy for visualization
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0  # Normalize as per LaneNet requirements

        # Run inference
        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: [image]}
        )

        # Post-process the results
        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
            with_lane_fit=True,
            data_source='tusimple'
        )
        mask_image = postprocess_result['mask_image']
        src_image = postprocess_result['source_image']
        num_lanes = len(postprocess_result['fit_params']) if postprocess_result['fit_params'] is not None else 0

        # Prepare binary_image and instance_image
        binary_image = np.array(binary_seg_image[0] * 255, dtype=np.uint8)  # Scale to 0-255
        binary_image = cv2.resize(binary_image, (image_vis.shape[1], image_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Use the clustered mask as the instance image (to match test_lanenet.py)
        instance_image = mask_image.copy()

        # Close the session
        sess.close()

        return {
            'src_image': src_image,
            'mask_image': mask_image,
            'instance_image': instance_image,
            'binary_image': binary_image,
            'num_lanes': num_lanes
        }

    else:  # Video processing
        output_paths = {
            'src_image': "processed_video_src.mp4",
            'mask_image': "processed_video_mask.mp4",
            'instance_image': "processed_video_instance.mp4",
            'binary_image': "processed_video_binary.mp4"
        }

        cap = cv2.VideoCapture(input_data)
        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Initialize video writers for each output
        writers = {
            key: cv2.VideoWriter(path, fourcc, fps, (frame_width, frame_height))
            for key, path in output_paths.items()
        }

        num_lanes_list = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process each frame
            frame_vis = frame.copy()
            frame = cv2.resize(frame, (512, 256), interpolation=cv2.INTER_LINEAR)
            frame = frame / 127.5 - 1.0

            # Run inference
            binary_seg_image, instance_seg_image = sess.run(
                [binary_seg_ret, instance_seg_ret],
                feed_dict={input_tensor: [frame]}
            )

            # Post-process
            postprocess_result = postprocessor.postprocess(
                binary_seg_result=binary_seg_image[0],
                instance_seg_result=instance_seg_image[0],
                source_image=frame_vis,
                with_lane_fit=True,
                data_source='tusimple'
            )
            mask_image = postprocess_result['mask_image']
            src_image = postprocess_result['source_image']
            num_lanes = len(postprocess_result['fit_params']) if postprocess_result['fit_params'] is not None else 0
            num_lanes_list.append(num_lanes)

            # Prepare binary_image and instance_image
            binary_image = np.array(binary_seg_image[0] * 255, dtype=np.uint8)
            binary_image = cv2.resize(binary_image, (frame_vis.shape[1], frame_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
            binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for video writing

            instance_image = mask_image.copy()

            # Write to respective videos
            writers['src_image'].write(src_image)
            writers['mask_image'].write(mask_image)
            writers['instance_image'].write(instance_image)
            writers['binary_image'].write(binary_image)

        cap.release()
        for writer in writers.values():
            writer.release()
        sess.close()

        # Compute the average number of lanes across frames
        avg_num_lanes = int(np.mean(num_lanes_list)) if num_lanes_list else 0

        return {
            'output_paths': output_paths,
            'num_lanes': avg_num_lanes
        }

# Streamlit app
st.title("Lane Detection System using LaneNet")

# Sidebar with model information
st.sidebar.header("About LaneNet")
st.sidebar.write("""
LaneNet is a deep learning model for lane detection, designed for autonomous driving applications.
It uses a segmentation-based approach to detect lanes in images and videos.
Learn more at the [GitHub repository](https://github.com/MaybeShewill-CV/lanenet-lane-detection.git).
""")

# Input section
st.header("Upload Your Input")
input_type = st.selectbox("Select input type:", ["Image", "Video"])
confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.9, 0.7)

# Specify the path to the pre-trained weights
weights_path = st.text_input("Path to Pre-trained Weights (.ckpt file)", value="lanenet-lane-detection/model/tusimple_lanenet/tusimple_lanenet.ckpt")

if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
else:
    uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

# Process the input
if uploaded_file is not None:
    try:
        if input_type == "Image":
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            st.image(image, caption="Original Image", use_column_width=True)

            if st.button("Detect Lanes"):
                start_time = time.time()
                result = detect_lanes(
                    input_data=image_np,
                    input_type="image",
                    confidence_threshold=confidence_threshold,
                    weights_path=weights_path
                )
                end_time = time.time()

                # Display all outputs
                st.subheader("Processed Outputs")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(result['src_image'], caption="Source Image with Lanes", use_column_width=True)
                with col2:
                    st.image(result['mask_image'], caption="Clustered Mask Image", use_column_width=True)

                col3, col4 = st.columns(2)
                with col3:
                    st.image(result['instance_image'], caption="Instance Segmentation Image", use_column_width=True)
                with col4:
                    st.image(result['binary_image'], caption="Binary Segmentation Image", use_column_width=True)

                st.write(f"Processing Time: {end_time - start_time:.2f} seconds")
                st.write(f"Number of Detected Lanes: {result['num_lanes']}")

        else:
            # Save the video temporarily
            video_path = "temp_video.mp4"
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())

            if st.button("Detect Lanes"):
                start_time = time.time()
                result = detect_lanes(
                    input_data=video_path,
                    input_type="video",
                    confidence_threshold=confidence_threshold,
                    weights_path=weights_path
                )
                output_paths = result['output_paths']
                end_time = time.time()

                # Display all outputs
                st.subheader("Original Video")
                st.video(video_path)

                st.subheader("Processed Outputs")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Source Video with Lanes:")
                    st.video(output_paths['src_image'])
                with col2:
                    st.write("Clustered Mask Video:")
                    st.video(output_paths['mask_image'])

                col3, col4 = st.columns(2)
                with col3:
                    st.write("Instance Segmentation Video:")
                    st.video(output_paths['instance_image'])
                with col4:
                    st.write("Binary Segmentation Video:")
                    st.video(output_paths['binary_image'])

                st.write(f"Processing Time: {end_time - start_time:.2f} seconds")
                st.write(f"Average Number of Detected Lanes: {result['num_lanes']}")

                # Provide download links for all processed videos
                st.subheader("Download Processed Videos")
                for key, path in output_paths.items():
                    with open(path, "rb") as f:
                        st.download_button(f"Download {key.replace('_', ' ').title()}", f, file_name=path)

                # Clean up temporary files
                os.remove(video_path)
                for path in output_paths.values():
                    os.remove(path)

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
else:
    st.write("Please upload a file to proceed.")