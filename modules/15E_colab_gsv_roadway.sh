# 15E - Colab GTF Object Detection Template (Roadway Images)
# Reference: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/object_detection.ipynb

image_urls = [
  # roadway1.jpg
  "https://drive.google.com/uc?export=view&id=1rxUnBwMI9PnIpAZdBc61FmQqoerNvNnc",
  # roadway2.jpg
  "https://drive.google.com/uc?export=view&id=1aWN-tg7GJeSHJ9uYkpcm4dT6kbGwDgD1",
  # roadway3.jpg
  "https://drive.google.com/uc?export=view&id=1Dipr9uzByfDfiOFvG20hJEV2cXmcZNPh",
  # roadway4.jpg
  "https://drive.google.com/uc?export=view&id=1m-Yo3W9z2XCDk4Ul-3-1sIXCOMvHLEiy",
  # roadway5.jpg
  "https://drive.google.com/uc?export=view&id=1rzToEf3TdNHVd6A_liSudu6JK1AgGC5Q",
  # roadway6.jpg
  "https://drive.google.com/uc?export=view&id=1Q15h_wpF4MmyzJl08B78e4AynVHwVMSe",
  # roadway7.jpg
  "https://drive.google.com/uc?export=view&id=18Zb0Tzs643Uf8YuV9JWjJwFNp0xJsVYX",
  # roadway8.jpg
  "https://drive.google.com/uc?export=view&id=1Nvzy0PNKOwdWCRqA43u06DxpdQHPgyNZ",
  # roadway9.jpg
  "https://drive.google.com/uc?export=view&id=1nc5vzOaRtoPLFwNWLZosWLC6T2FPu8hK",
  # roadway10.jpg
  "https://drive.google.com/uc?export=view&id=1VOTDtQW6YaaQNpI4Bf2OEa-NeUCEQFjY"
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
