{-# LANGUAGE GADTs
           , DataKinds
           , TypeApplications
           , ScopedTypeVariables
           , ExplicitForAll
           , AllowAmbiguousTypes
           , KindSignatures
           , TypeOperators
           , RankNTypes
           , GeneralizedNewtypeDeriving #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

module Lib where


import Numeric.LinearAlgebra hiding (Vector, Matrix)
import qualified Numeric.LinearAlgebra as LA
import qualified Data.Vector.Storable as VS
import GHC.TypeLits
import Data.Proxy
import System.Random
import Control.Monad
import Numeric.AD

newtype Vector (n :: Nat) = Vector { _vec :: VS.Vector Float }
  deriving (Show, Floating)

instance KnownNat n => Num (Vector n) where
  (Vector v1) + (Vector v2) = Vector $ v1 + v2
  (Vector v1) - (Vector v2) = Vector $ v1 - v2
  (Vector v1) * (Vector v2) = Vector $ v1 * v2
  abs                       = vectorMap abs
  fromInteger               = vecFromList . (:[]) . fromInteger
  negate                    = vectorMap negate
  signum                    = vectorMap signum

instance KnownNat n => Fractional (Vector n) where
  (Vector v1) / (Vector v2) = Vector $ v1 / v2
  fromRational              = vecFromList . (:[]) . fromRational


natToInt :: forall n. KnownNat n => Int
natToInt = fromIntegral . natVal $ Proxy @n

vecFromList :: forall n. KnownNat n => [Float] -> Vector n
vecFromList = Vector . VS.fromList . take (natToInt @n) . cycle

vecLength :: forall n. KnownNat n => Vector n -> Int
vecLength _ = natToInt @n

vecTail :: forall n. KnownNat n => Vector (n+1) -> Vector n
vecTail (Vector v) = Vector $ VS.tail v

newtype Matrix (n :: Nat) (m :: Nat) = Matrix { _mat :: LA.Matrix Float }
  deriving Show

matrixWidth :: forall n m. (KnownNat n, KnownNat m) => Matrix n m -> Int
matrixWidth _ = natToInt @n

matrixHeight :: forall n m. (KnownNat n, KnownNat m) => Matrix n m -> Int
matrixHeight _ = natToInt @m

matFromList :: forall n m. (KnownNat n, KnownNat m) => [Float] -> Matrix n m
matFromList = Matrix . (h><w) . cycle
  where w = natToInt @n
        h = natToInt @m

randomMatrix :: forall n m. (KnownNat n, KnownNat m) => IO (Matrix n m)
randomMatrix = matFromList <$> replicateM (w*h) (randomRIO (0.001, 0.1))
    where w = natToInt @n
          h = natToInt @m

data Network :: Nat -> Nat -> * where
  Layer :: forall m n. (KnownNat n, KnownNat m) =>
                       Matrix (n+1) m ->
                       (forall a. Floating a => a -> a) ->
                       Network n m
  Composition :: (KnownNat n, KnownNat m, KnownNat k) =>
                 Network m k ->
                 Network n m ->
                 Network n k


conductSignal :: forall n m. (KnownNat n, KnownNat m) =>
                               Network n m ->
                               Vector n ->
                               Vector m
conductSignal (Layer mat f) = vectorMap f . mvMul mat . vectorCons 1
conductSignal (Composition network2 network1) = conductSignal network2 . conductSignal network1

vectorMap :: (Float -> Float) -> Vector n -> Vector n
vectorMap f (Vector v) = Vector $ VS.map f v

mvMul :: Matrix n m -> Vector n -> Vector m
mvMul (Matrix m) (Vector v) = Vector $ m #> v

vectorCons :: KnownNat n => Float -> Vector n -> Vector (n+1)
vectorCons n (Vector v) = Vector $ VS.cons n v

randomLayer :: forall n m. (KnownNat n, KnownNat m) => (forall a. Floating a => a -> a) -> IO (Network n m)
randomLayer f = (`Layer` f) <$> randomMatrix

scaleMatrix :: (KnownNat n, KnownNat m) => Float -> Matrix m n -> Matrix m n
scaleMatrix s (Matrix m) = Matrix $ LA.scale s m

transposeMat :: (KnownNat n, KnownNat m) => Matrix n m -> Matrix m n
transposeMat (Matrix mat) = Matrix $ LA.tr mat

outerProduct :: (KnownNat n, KnownNat m) => Vector n -> Vector m -> Matrix n m
outerProduct (Vector v1) (Vector v2) = Matrix $ v2 `outer` v1

vvMul :: (KnownNat n) => Vector n -> Vector n -> Vector n
vvMul (Vector v1) (Vector v2) = Vector $ v1 * v2

matrixSub :: (KnownNat m, KnownNat n) => Matrix n m -> Matrix n m -> Matrix n m
matrixSub (Matrix m1) (Matrix m2) = Matrix $ m1 - m2


useTrainingExample :: forall m n. (KnownNat n, KnownNat m) =>
                        Vector m ->
                        Network n m ->
                        Vector n ->
                        (Network n m, Vector n)
useTrainingExample diffLoss (Layer weights activation) input = (trainedLayer, inputGradient)
  where
    trainedLayer = Layer newWeights activation
    newWeights :: Matrix (n+1) m
    newWeights = matrixSub weights $ scaleMatrix 0.01 weightGradient
    consedInput = vectorCons 1 input
    preActivationOutput = mvMul weights consedInput
    output = vectorMap activation preActivationOutput
    diffActivation = vectorMap (diff activation) preActivationOutput
    weightGradient = outerProduct consedInput (diffLoss * diffActivation)
    inputGradient  = vecTail $ mvMul (transposeMat weights) (diffLoss * diffActivation)
useTrainingExample diffLoss (Composition network2 network1) input =
    (Composition trainedNet2 trainedNet1, inputGradient)
  where (trainedNet2,outputGrad1)   = useTrainingExample diffLoss network2 input2
        input2                      = conductSignal network1 input
        (trainedNet1,inputGradient) = useTrainingExample outputGrad1 network1 input

showExample :: (KnownNat n, KnownNat m) =>
                (Vector m -> Vector m -> Vector m) ->
                Network n m ->
                (Vector n, Vector m) ->
                Network n m
showExample lossGrad network (input,expected) = fst $ useTrainingExample diffLoss network input
  where diffLoss = lossGrad expected output
        output   = conductSignal network input


sigmoid input = 1/(1+exp(-input))
                -- (1/) . (1+) . exp . negate

epsilon :: Floating a => a
epsilon = 0.0000001

logLoss :: Floating a => [a] -> a
logLoss [expected, output] = negate $ expected * log (output + epsilon) + (1 - expected) * log (1 - output + epsilon)

gradLogLoss expected output = grad logLoss [expected, output] !! 1

-- "mnist/train-set"

getTrainSet :: IO [(Vector 784, Vector 10)]
getTrainSet =
  map (\(a,b) -> (vecFromList a, toOneHot b)) .
  map (read @([Float],Int)) .
  lines <$> readFile  "mnist/train-set"
  where
    toOneHot n = vecFromList $ map (\x -> if x == n then 1 else 0) [0..]

eval network (input,expected) = do
  putStrLn "EXPECTED:"
  print expected
  putStrLn "PREDICTED:"
  print $ conductSignal network input

trainTest = do
  trainSet <- take 100 <$> getTrainSet
  untrainedNetwork <- randomLayer sigmoid
  let trainedNetwork = foldl (useTrainingExample gradLogLoss) untrainedNetwork trainSet
      evalSet = take 10 trainSet
  mapM_ (eval trainedNetwork) evalSet

{-

  diff (loss . activation . mvMul(weights, input))

   => diff loss (activation . mvMul (weights,input)) * diff (activation . mvMul (weights,input))

   => diff loss (output) * diff activation (mvMul (weights,input)) * diff mvMul(weights,input)
-}
