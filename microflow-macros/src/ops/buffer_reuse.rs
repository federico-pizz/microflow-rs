//! Buffer-reuse (ping-pong) code generation support.
//!
//! The default MicroFlow codegen emits each layer as a shadowing
//! `let input = op(input, ...)` statement. Because every statement has a
//! distinct const-generic tensor type, LLVM gives each one its own stack slot
//! and never reuses the dead ones, so `predict_inner`'s peak stack grows with
//! the *sum* of all intermediate tensors (the "stack wall").
//!
//! When the `buffer-reuse` feature is enabled and the model is an eligible
//! linear chain (see `lib.rs`), the compiler instead emits two persistent
//! enum buffers and ping-pongs between them: each step fills `out` (a fresh
//! local of the destination shape) from `src` via the op's in-place `*_into`
//! variant, then re-tags the destination enum. Host experiments showed this
//! collapses the peak to ~2×MAX (two live buffers) instead of the sum, which
//! is what unblocks dual-core inference on the ESP32-S3 stack.
//!
//! This trait lets each into-able op contribute the two op-specific pieces the
//! generic ping-pong driver needs: the hoisted `const` (filters/weights) and
//! the in-place call itself.

use proc_macro2::{Ident, TokenStream as TokenStream2};
use quote::{format_ident, quote};

use crate::ops::conv_2d::TokenConv2D;
use crate::ops::depthwise_conv_2d::TokenDepthwiseConv2D;
use crate::ops::max_pool_2d::TokenMaxPool2D;
use crate::ops::resize_nearest_neighbor::TokenResizeNearestNeighbor;
use crate::quantize::TokenQuantized;

/// Implemented by the Tensor4D→Tensor4D ops that expose an in-place `*_into`
/// runtime variant and can therefore participate in the ping-pong chain.
pub(crate) trait BufferReuse {
    /// `const` items (e.g. filters/weights) hoisted to `predict_inner` scope so
    /// the generated step functions can reference them. Empty by default.
    fn br_consts(&self) -> TokenStream2 {
        TokenStream2::new()
    }

    /// The in-place call `microflow::ops::NAME_into(#out, #src, ...);`,
    /// where `#src` is an `&Tensor4D` (already a reference) and `#out` is a
    /// caller-owned output tensor to fill.
    fn br_call(&self, src: &Ident, out: &Ident) -> TokenStream2;

    /// Output tensor shape `[1, ROWS, COLS, CHANS]`, used to type the
    /// destination enum variant.
    fn br_out_shape(&self) -> Vec<usize>;
}

impl<T: TokenQuantized> BufferReuse for TokenConv2D<T> {
    fn br_consts(&self) -> TokenStream2 {
        let filters_ident = format_ident!("filters_{}", self.index);
        let filters_type = self.filters.type_tokens();
        let filters = &self.filters;
        quote! { const #filters_ident: #filters_type = #filters; }
    }

    fn br_call(&self, src: &Ident, out: &Ident) -> TokenStream2 {
        let filters_ident = format_ident!("filters_{}", self.index);
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let view_padding = self.view_padding;
        let (strides_0, strides_1) = self.strides;
        let (constants_0, constants_1, constants_2, constants_3) = &self.constants;
        quote! {
            microflow::ops::conv_2d_into(
                #out,
                #src,
                &#filters_ident,
                [#(#output_scale),*],
                [#(#output_zero_point),*],
                microflow::ops::Conv2DOptions {
                    fused_activation: #fused_activation,
                    view_padding: #view_padding,
                    strides: (#strides_0, #strides_1),
                },
                (#constants_0, #constants_1, #constants_2, #constants_3)
            );
        }
    }

    fn br_out_shape(&self) -> Vec<usize> {
        self.output.shape.clone()
    }
}

impl<T: TokenQuantized> BufferReuse for TokenDepthwiseConv2D<T> {
    fn br_consts(&self) -> TokenStream2 {
        let weights_ident = format_ident!("weights_{}", self.index);
        let weights_type = self.weights.type_tokens();
        let weights = &self.weights;
        quote! { const #weights_ident: #weights_type = #weights; }
    }

    fn br_call(&self, src: &Ident, out: &Ident) -> TokenStream2 {
        let weights_ident = format_ident!("weights_{}", self.index);
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let view_padding = self.view_padding;
        let (strides_0, strides_1) = self.strides;
        let (constants_0, constants_1, constants_2, constants_3) = &self.constants;
        quote! {
            microflow::ops::depthwise_conv_2d_into(
                #out,
                #src,
                &#weights_ident,
                [#(#output_scale),*],
                [#(#output_zero_point),*],
                microflow::ops::DepthwiseConv2DOptions {
                    fused_activation: #fused_activation,
                    view_padding: #view_padding,
                    strides: (#strides_0, #strides_1),
                },
                (#constants_0, #constants_1, #constants_2, #constants_3)
            );
        }
    }

    fn br_out_shape(&self) -> Vec<usize> {
        self.output.shape.clone()
    }
}

impl<T: TokenQuantized> BufferReuse for TokenMaxPool2D<T> {
    fn br_call(&self, src: &Ident, out: &Ident) -> TokenStream2 {
        let (filter_shape_0, filter_shape_1) = self.filter_shape;
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let fused_activation = self.fused_activation;
        let view_padding = self.view_padding;
        let (strides_0, strides_1) = self.strides;
        let (constants_0, constants_1) = self.constants;
        quote! {
            microflow::ops::max_pool_2d_into(
                #out,
                #src,
                (nalgebra::Const::<#filter_shape_0>, nalgebra::Const::<#filter_shape_1>),
                [#(#output_scale),*],
                [#(#output_zero_point),*],
                microflow::ops::MaxPool2DOptions {
                    fused_activation: #fused_activation,
                    view_padding: #view_padding,
                    strides: (#strides_0, #strides_1),
                },
                (#constants_0, #constants_1)
            );
        }
    }

    fn br_out_shape(&self) -> Vec<usize> {
        self.output.shape.clone()
    }
}

impl<T: TokenQuantized> BufferReuse for TokenResizeNearestNeighbor<T> {
    fn br_call(&self, src: &Ident, out: &Ident) -> TokenStream2 {
        let output_scale = &self.output.scale;
        let output_zero_point = &self.output.zero_point;
        let align_corners = self.align_corners;
        let half_pixel_centers = self.half_pixel_centers;
        let (constants_0, constants_1) = self.constants;
        quote! {
            microflow::ops::resize_nearest_neighbor_into(
                #out,
                #src,
                [#(#output_scale),*],
                [#(#output_zero_point),*],
                microflow::ops::ResizeNearestNeighborOptions {
                    align_corners: #align_corners,
                    half_pixel_centers: #half_pixel_centers,
                },
                (#constants_0, #constants_1),
            );
        }
    }

    fn br_out_shape(&self) -> Vec<usize> {
        self.output.shape.clone()
    }
}
