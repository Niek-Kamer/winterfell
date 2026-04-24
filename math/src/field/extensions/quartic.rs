// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// Quartic extension. Added in the Niek-Kamer/winterfell fork so that 31-bit
// STARK-friendly fields (BabyBear) can reach 95+ bit conjectured soundness.

use alloc::string::{String, ToString};
use core::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    slice,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use utils::{
    AsBytes, ByteReader, ByteWriter, Deserializable, DeserializationError, Randomizable,
    Serializable, SliceReader,
};

use super::{ExtensibleField, ExtensionOf, FieldElement};

// QUARTIC EXTENSION FIELD
// ================================================================================================

/// Represents an element in a quartic extension of a [StarkField](crate::StarkField).
///
/// The extension element is defined as α + β * φ + γ * φ^2 + δ * φ^3, where φ is a root of an
/// irreducible polynomial defined by the implementation of the [ExtensibleField] trait, and α,
/// β, γ, δ are base field elements.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct QuartExtension<B: ExtensibleField<4>>(B, B, B, B);

impl<B: ExtensibleField<4>> QuartExtension<B> {
    /// Returns a new extension element instantiated from the provided base elements.
    pub const fn new(a: B, b: B, c: B, d: B) -> Self {
        Self(a, b, c, d)
    }

    /// Returns true if the base field specified by B type parameter supports quartic extensions.
    pub fn is_supported() -> bool {
        <B as ExtensibleField<4>>::is_supported()
    }

    /// Returns an array of base field elements comprising this extension field element.
    ///
    /// The order of base elements in the returned array is the same as the order in which
    /// the elements are provided to the [QuartExtension::new()] constructor.
    pub const fn to_base_elements(self) -> [B; 4] {
        [self.0, self.1, self.2, self.3]
    }
}

impl<B: ExtensibleField<4>> FieldElement for QuartExtension<B> {
    type PositiveInteger = B::PositiveInteger;
    type BaseField = B;

    const EXTENSION_DEGREE: usize = 4;

    const ELEMENT_BYTES: usize = B::ELEMENT_BYTES * Self::EXTENSION_DEGREE;
    const IS_CANONICAL: bool = B::IS_CANONICAL;
    const ZERO: Self = Self(B::ZERO, B::ZERO, B::ZERO, B::ZERO);
    const ONE: Self = Self(B::ONE, B::ZERO, B::ZERO, B::ZERO);

    // ALGEBRA
    // --------------------------------------------------------------------------------------------

    #[inline]
    fn double(self) -> Self {
        Self(self.0.double(), self.1.double(), self.2.double(), self.3.double())
    }

    #[inline]
    fn square(self) -> Self {
        let a = <B as ExtensibleField<4>>::square([self.0, self.1, self.2, self.3]);
        Self(a[0], a[1], a[2], a[3])
    }

    #[inline]
    fn inv(self) -> Self {
        if self == Self::ZERO {
            return self;
        }

        // For a degree-4 extension the norm N(x) = x * σ(x) * σ²(x) * σ³(x) lies in the base
        // field, and x^-1 = σ(x) * σ²(x) * σ³(x) / N(x). This mirrors the cubic case with one
        // extra Frobenius step.
        let x = [self.0, self.1, self.2, self.3];
        let c1 = <B as ExtensibleField<4>>::frobenius(x);
        let c2 = <B as ExtensibleField<4>>::frobenius(c1);
        let c3 = <B as ExtensibleField<4>>::frobenius(c2);
        let numerator = <B as ExtensibleField<4>>::mul(
            <B as ExtensibleField<4>>::mul(c1, c2),
            c3,
        );

        let norm = <B as ExtensibleField<4>>::mul(x, numerator);
        debug_assert_eq!(norm[1], B::ZERO, "norm must be in the base field");
        debug_assert_eq!(norm[2], B::ZERO, "norm must be in the base field");
        debug_assert_eq!(norm[3], B::ZERO, "norm must be in the base field");
        let denom_inv = norm[0].inv();

        Self(
            numerator[0] * denom_inv,
            numerator[1] * denom_inv,
            numerator[2] * denom_inv,
            numerator[3] * denom_inv,
        )
    }

    #[inline]
    fn conjugate(&self) -> Self {
        let result = <B as ExtensibleField<4>>::frobenius([self.0, self.1, self.2, self.3]);
        Self(result[0], result[1], result[2], result[3])
    }

    // BASE ELEMENT CONVERSIONS
    // --------------------------------------------------------------------------------------------

    fn base_element(&self, i: usize) -> Self::BaseField {
        match i {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            3 => self.3,
            _ => panic!("element index must be smaller than 4, but was {i}"),
        }
    }

    fn slice_as_base_elements(elements: &[Self]) -> &[Self::BaseField] {
        let ptr = elements.as_ptr();
        let len = elements.len() * Self::EXTENSION_DEGREE;
        unsafe { slice::from_raw_parts(ptr as *const Self::BaseField, len) }
    }

    fn slice_from_base_elements(elements: &[Self::BaseField]) -> &[Self] {
        assert!(
            elements.len().is_multiple_of(Self::EXTENSION_DEGREE),
            "number of base elements must be divisible by 4, but was {}",
            elements.len()
        );

        let ptr = elements.as_ptr();
        let len = elements.len() / Self::EXTENSION_DEGREE;
        unsafe { slice::from_raw_parts(ptr as *const Self, len) }
    }

    // SERIALIZATION / DESERIALIZATION
    // --------------------------------------------------------------------------------------------

    fn elements_as_bytes(elements: &[Self]) -> &[u8] {
        unsafe {
            slice::from_raw_parts(
                elements.as_ptr() as *const u8,
                elements.len() * Self::ELEMENT_BYTES,
            )
        }
    }

    unsafe fn bytes_as_elements(bytes: &[u8]) -> Result<&[Self], DeserializationError> {
        if !bytes.len().is_multiple_of(Self::ELEMENT_BYTES) {
            return Err(DeserializationError::InvalidValue(format!(
                "number of bytes ({}) does not divide into whole number of field elements",
                bytes.len(),
            )));
        }

        let p = bytes.as_ptr();
        let len = bytes.len() / Self::ELEMENT_BYTES;

        // make sure the bytes are aligned on the boundary consistent with base element alignment
        if !(p as usize).is_multiple_of(Self::BaseField::ELEMENT_BYTES) {
            return Err(DeserializationError::InvalidValue(
                "slice memory alignment is not valid for this field element type".to_string(),
            ));
        }

        Ok(slice::from_raw_parts(p as *const Self, len))
    }
}

impl<B: ExtensibleField<4>> ExtensionOf<B> for QuartExtension<B> {
    #[inline(always)]
    fn mul_base(self, other: B) -> Self {
        let result =
            <B as ExtensibleField<4>>::mul_base([self.0, self.1, self.2, self.3], other);
        Self(result[0], result[1], result[2], result[3])
    }
}

impl<B: ExtensibleField<4>> Randomizable for QuartExtension<B> {
    const VALUE_SIZE: usize = Self::ELEMENT_BYTES;

    fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
        Self::try_from(bytes).ok()
    }
}

impl<B: ExtensibleField<4>> fmt::Display for QuartExtension<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {}, {}, {})", self.0, self.1, self.2, self.3)
    }
}

// OVERLOADED OPERATORS
// ------------------------------------------------------------------------------------------------

impl<B: ExtensibleField<4>> Add for QuartExtension<B> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2, self.3 + rhs.3)
    }
}

impl<B: ExtensibleField<4>> AddAssign for QuartExtension<B> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}

impl<B: ExtensibleField<4>> Sub for QuartExtension<B> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2, self.3 - rhs.3)
    }
}

impl<B: ExtensibleField<4>> SubAssign for QuartExtension<B> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<B: ExtensibleField<4>> Mul for QuartExtension<B> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let result = <B as ExtensibleField<4>>::mul(
            [self.0, self.1, self.2, self.3],
            [rhs.0, rhs.1, rhs.2, rhs.3],
        );
        Self(result[0], result[1], result[2], result[3])
    }
}

impl<B: ExtensibleField<4>> MulAssign for QuartExtension<B> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs
    }
}

impl<B: ExtensibleField<4>> Div for QuartExtension<B> {
    type Output = Self;

    #[inline]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inv()
    }
}

impl<B: ExtensibleField<4>> DivAssign for QuartExtension<B> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs
    }
}

impl<B: ExtensibleField<4>> Neg for QuartExtension<B> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self(-self.0, -self.1, -self.2, -self.3)
    }
}

// TYPE CONVERSIONS
// ------------------------------------------------------------------------------------------------

impl<B: ExtensibleField<4>> From<B> for QuartExtension<B> {
    fn from(value: B) -> Self {
        Self(value, B::ZERO, B::ZERO, B::ZERO)
    }
}

impl<B: ExtensibleField<4>> From<u32> for QuartExtension<B> {
    fn from(value: u32) -> Self {
        Self(B::from(value), B::ZERO, B::ZERO, B::ZERO)
    }
}

impl<B: ExtensibleField<4>> From<u16> for QuartExtension<B> {
    fn from(value: u16) -> Self {
        Self(B::from(value), B::ZERO, B::ZERO, B::ZERO)
    }
}

impl<B: ExtensibleField<4>> From<u8> for QuartExtension<B> {
    fn from(value: u8) -> Self {
        Self(B::from(value), B::ZERO, B::ZERO, B::ZERO)
    }
}

impl<B: ExtensibleField<4>> TryFrom<u64> for QuartExtension<B> {
    type Error = String;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        match B::try_from(value) {
            Ok(elem) => Ok(Self::from(elem)),
            Err(_) => Err(format!(
                "invalid field element: value {value} is greater than or equal to the field modulus"
            )),
        }
    }
}

impl<B: ExtensibleField<4>> TryFrom<u128> for QuartExtension<B> {
    type Error = String;

    fn try_from(value: u128) -> Result<Self, Self::Error> {
        match B::try_from(value) {
            Ok(elem) => Ok(Self::from(elem)),
            Err(_) => Err(format!(
                "invalid field element: value {value} is greater than or equal to the field modulus"
            )),
        }
    }
}

impl<B: ExtensibleField<4>> TryFrom<&'_ [u8]> for QuartExtension<B> {
    type Error = DeserializationError;

    /// Converts a slice of bytes into a field element; returns error if the value encoded in bytes
    /// is not a valid field element. The bytes are assumed to be in little-endian byte order.
    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        if bytes.len() < Self::ELEMENT_BYTES {
            return Err(DeserializationError::InvalidValue(format!(
                "not enough bytes for a full field element; expected {} bytes, but was {} bytes",
                Self::ELEMENT_BYTES,
                bytes.len(),
            )));
        }
        if bytes.len() > Self::ELEMENT_BYTES {
            return Err(DeserializationError::InvalidValue(format!(
                "too many bytes for a field element; expected {} bytes, but was {} bytes",
                Self::ELEMENT_BYTES,
                bytes.len(),
            )));
        }
        let mut reader = SliceReader::new(bytes);
        Self::read_from(&mut reader)
    }
}

impl<B: ExtensibleField<4>> AsBytes for QuartExtension<B> {
    fn as_bytes(&self) -> &[u8] {
        // TODO: take endianness into account
        let self_ptr: *const Self = self;
        unsafe { slice::from_raw_parts(self_ptr as *const u8, Self::ELEMENT_BYTES) }
    }
}

// SERIALIZATION / DESERIALIZATION
// ------------------------------------------------------------------------------------------------

impl<B: ExtensibleField<4>> Serializable for QuartExtension<B> {
    fn write_into<W: ByteWriter>(&self, target: &mut W) {
        self.0.write_into(target);
        self.1.write_into(target);
        self.2.write_into(target);
        self.3.write_into(target);
    }
}

impl<B: ExtensibleField<4>> Deserializable for QuartExtension<B> {
    fn read_from<R: ByteReader>(source: &mut R) -> Result<Self, DeserializationError> {
        let value0 = B::read_from(source)?;
        let value1 = B::read_from(source)?;
        let value2 = B::read_from(source)?;
        let value3 = B::read_from(source)?;
        Ok(Self(value0, value1, value2, value3))
    }
}
