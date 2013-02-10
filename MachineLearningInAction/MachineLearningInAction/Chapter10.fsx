// Chapter 10: K-means clustering
#load "KMeansClustering.fs"
open MachineLearning.KMeansClustering

// test data
let rng = new System.Random()
let centroids = [ [| 0.; 0. |]; [| 20.; 30. |]; [| -40.; -60. |] ]
// Create 50 points centered around each Centroid
let data = [ 
    for centroid in centroids do
        for i in 1 .. 50 -> 
            Array.map (fun x -> x + 5. * (rng.NextDouble() - 0.5)) centroid ]

let factory = randomCentroids<float[]> rng
let search = kmeans euclidean factory avgCentroid data 3
let identifiedCentroids = fst search |> Seq.toList