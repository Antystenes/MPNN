{-# LANGUAGE GADTs
           , DataKinds
           , TypeApplications
           , ScopedTypeVariables
           , ExplicitForAll
           , AllowAmbiguousTypes
           , KindSignatures
           , TypeOperators
           , RankNTypes #-}
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
  deriving Show

natToInt :: forall n. KnownNat n => Int
natToInt = fromIntegral . natVal $ Proxy @n

vecFromList :: forall n. KnownNat n => [Float] -> Vector n
vecFromList = Vector . VS.fromList . take (natToInt @n) . cycle

vecLength :: forall n. KnownNat n => Vector n -> Int
vecLength _ = natToInt @n

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
                       (forall a. Num a => a -> a) ->
                       Network n m

vectorMap :: (Float -> Float) -> Vector n -> Vector n
vectorMap f (Vector v) = Vector $ VS.map f v

mvMul :: Matrix n m -> Vector n -> Vector m
mvMul (Matrix m) (Vector v) = Vector $ m #> v

vectorCons :: KnownNat n => Float -> Vector n -> Vector (n+1)
vectorCons n (Vector v) = Vector $ VS.cons n v

conductSignal :: forall n m. (KnownNat n, KnownNat m) =>
                               Network n m ->
                               Vector n ->
                               Vector m
conductSignal (Layer mat f) = vectorMap f . mvMul mat . vectorCons 1

randomLayer :: forall n m. (KnownNat n, KnownNat m) => (forall a. Num a => a -> a) -> IO (Network n m)
randomLayer f = (`Layer` f) <$> randomMatrix

scaleMatrix :: (KnownNat n, KnownNat m) => Float -> Matrix m n -> Matrix m n
scaleMatrix s (Matrix m) = Matrix $ LA.scale s m

transposeMat :: (KnownNat n, KnownNat m) => Matrix n m -> Matrix m n
transposeMat (Matrix mat) = Matrix $ LA.tr mat

outerProduct :: (KnownNat n, KnownNat m) => Vector n -> Vector m -> Matrix n m
outerProduct (Vector v1) (Vector v2) = Matrix $ v1 `outer` v2

vvMul :: (KnownNat n) => Vector n -> Vector n -> Vector n
vvMul (Vector v1) (Vector v2) = Vector $ v1 * v2

matrixSub :: (KnownNat m, KnownNat n) => Matrix n m -> Matrix n m -> Matrix n m
matrixSub (Matrix m1) (Matrix m2) = Matrix $ m1 - m2

useTrainingExample :: forall m n. (KnownNat n, KnownNat m) =>
                        (Vector m -> Vector m -> Vector m) ->
                        Network n m ->
                        (Vector n, Vector m) ->
                        Network n m
useTrainingExample lossGradient (Layer weights activation) (input, expected) = trainedLayer
  where
    trainedLayer = Layer newWeights activation
    newWeights :: Matrix (n+1) m
    newWeights = matrixSub weights $ scaleMatrix 0.01 weightGradient
    output = conductSignal trainedLayer input
    outputGradient = lossGradient output expected
    activationDerivative = diff activation
    weightGradient = outerProduct (vectorCons 1 input) .
                     vvMul outputGradient .
                     vectorMap activationDerivative .
                     mvMul weights .
                     vectorCons 1 $ input
