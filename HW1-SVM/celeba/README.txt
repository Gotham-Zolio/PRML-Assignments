========================
File Information
========================

- celeba_filtered/xxxxxx.jpg
    300 aligned face images belonging to 10 identities, with 30 images for each identity

- celeba_filtered.txt
    First Row (Headers): Identity (10 different identities in total), Filename (corresponding to the celeba_filtered folder), Names of 40 binary attributes (including gender)
    The Rest of the Rows: Arranged according to the first row, "1" represents positive while "-1" represents negative

- celeba_landmarks.txt
    First Row (Headers): Filename (corresponding to the celeba_filtered folder), Names of 10 landmarks
    The Rest of the Rows: Arranged according to the first row. Numbers are in the pixel coordinate.

- partition.txt
    image ids for training and testing sets where "0" represents training image, "1" represents testing image.