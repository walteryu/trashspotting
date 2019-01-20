# 15 - Colab GTF Object Detection Template
# Reference: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb

image_urls = [
  "https://cdn.pixabay.com/photo/2016/03/16/14/12/garbage-can-1260832_960_720.jpg",
  "https://cdn.pixabay.com/photo/2017/09/08/18/20/garbage-2729608_960_720.jpg",
  "https://cdn.pixabay.com/photo/2013/07/05/12/20/rubbish-143465_960_720.jpg",
  "https://cdn.pixabay.com/photo/2016/03/16/14/12/garbage-1260833_960_720.jpg",
  "https://cdn.pixabay.com/photo/2019/01/11/02/39/coffeetogo-3926395_960_720.jpg"
]

for image_url in image_urls:
  image_path = download_and_resize_image(image_url, 640, 480)
  with tf.gfile.Open(image_path, "rb") as binfile:
    image_string = binfile.read()

  inference_start_time = time.clock()
  result_out, image_out = session.run(
      [result, decoded_image],
      feed_dict={image_string_placeholder: image_string})
  print("Found %d objects." % len(result_out["detection_scores"]))
  print("Inference took %.2f seconds." % (time.clock()-inference_start_time))

  image_with_boxes = draw_boxes(
    np.array(image_out), result_out["detection_boxes"],
    result_out["detection_class_entities"], result_out["detection_scores"])

  display_image(image_with_boxes)
