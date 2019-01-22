# 15D - Colab GTF Object Detection Template (Litter Images)
# Reference: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb

image_urls = [
  # trash1.jpg
  "https://drive.google.com/uc?export=view&id=1WBDwpcSfGD93JfB0-S0YkbRwHXptp-yb",
  # trash2.jpg
  "https://drive.google.com/uc?export=view&id=1s-nYGQoqYFwmDJT1PwzMzPhpmc5ShFHo",
  # trash3.jpg
  "https://drive.google.com/uc?export=view&id=1WsRMHasZHpAaQ0PkcV2AJSHxDT_JKQiO",
  # trash4.jpg
  "https://drive.google.com/uc?export=view&id=1gE7yTHXXgDI41iXBDDdws64O4gWdELsr",
  # trash5.jpg
  "https://drive.google.com/uc?export=view&id=1ptJ3_jj-nKAqGLUtW1cTctUGtrX1_27N",
  # trash6.jpg
  "https://drive.google.com/uc?export=view&id=18MR1gggx9WroHReXTKmTlOEj1C_kX3Zk",
  # trash7.jpg
  "https://drive.google.com/uc?export=view&id=1Rw9tmoDkGbPHdFnO_POU9SWcPOIhwRbb",
  # trash8.jpg
  "https://drive.google.com/uc?export=view&id=1W3hLsjZUd_IvOsOoy1sgSipD0dir8Lsn",
  # trash9.jpg
  "https://drive.google.com/uc?export=view&id=1c-ev_ifZ5UVqzAK2x8oe4nglBy8g4Of2",
  # trash10.jpg
  "https://drive.google.com/uc?export=view&id=1zlizpaV9NxzC6jO8dCnOA1AU9pTNyEQd"
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
