namespace KMeansClustering.Tests

open MachineLearning.KMeansClustering
open NUnit.Framework
open FsUnit

[<TestFixture>]
type ``K-means clustering tests`` () =

    [<Test>]
    member this.``closest verification`` () =
        let centroids = [ 0.; 1. ]
        let distance (x: float) y = abs (x-y)
        
        closest distance centroids 0. |> should equal (0, 0.)
        closest distance centroids 1. |> should equal (1, 0.)
        closest distance centroids 5. |> should equal (1, 4.)
        closest distance centroids 0.1 |> should equal (0, 0.1)

    [<Test>]
    member this.``Euclidean distance verification`` () =
        let x = [| 0.; 0. |]
        let y = [| 3.; 4. |]
        euclidean x y |> should equal 5.

    [<Test>]
    member this.``avgCentroid verification`` () =
        let sample = [ [| 0.; 0. |]; [| 2.; 4. |] ] 
        let current = [| 1.; 1. |]
        avgCentroid current sample |> should equal [| 1.; 2. |]
        avgCentroid current [] |> should equal current

