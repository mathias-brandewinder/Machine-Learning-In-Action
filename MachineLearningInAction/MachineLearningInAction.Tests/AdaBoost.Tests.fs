namespace AdaBoost.Tests

open MachineLearning.AdaBoost
open NUnit.Framework
open FsUnit

[<TestFixture>]
type ``AdaBoost tests`` () =

    [<Test>]
    member this.``normalize verification`` () =

        let v = [| 1.0; 3.0; 4.0 |]
        normalize v |> should equal [| 0.125; 0.375; 0.500|]

    [<Test>]
    member this.``stumpClassify verification`` () =

        let obs = [| 1.0; 2.0 |]
        // obs.[0] = 1.0 is >= 2.0        
        stumpClassify 0 2.0 (>=) obs |> should equal -1.0
        // obs.[0] = 1.0 is not >= 1.0        
        stumpClassify 0 1.0 (>=) obs |> should equal 1.0
        // obs.[1] = 2.0 is <= 3.0        
        stumpClassify 1 3.0 (<=) obs |> should equal 1.0
        // obs.[1] = 2.0 is not <= 1.0        
        stumpClassify 1 1.0 (<=) obs |> should equal -1.0

    [<Test>]
    member this.``classify verification`` () =

        let obs = [| 1.0; 2.0 |]

        let model = 
            [ { Alpha = 1.0; Classifier = fun x -> 1.0 };
              { Alpha = 0.5; Classifier = fun x -> -1.0 } ]
        // majority vote of weak learners is 1.0
        classify model obs |> should equal 1.0

        let model = 
            [ { Alpha = 1.0; Classifier = fun x -> -1.0 };
              { Alpha = 0.5; Classifier = fun x -> 1.0 } ]
        // majority vote of weak learners is -1.0
        classify model obs |> should equal -1.0
