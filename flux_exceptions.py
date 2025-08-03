#!/usr/bin/env python3
"""
FLUX.1 Krea Custom Exception Classes
Specific error types for better error handling and user feedback
"""

import torch

class FluxBaseException(Exception):
    """Base exception for all FLUX-related errors"""
    pass

class ModelNotFoundError(FluxBaseException):
    """Raised when required model files are not found"""
    def __init__(self, model_name, expected_path=None):
        self.model_name = model_name
        self.expected_path = expected_path
        message = f"Model '{model_name}' not found"
        if expected_path:
            message += f" at path: {expected_path}"
        super().__init__(message)

class AuthenticationError(FluxBaseException):
    """Raised when HuggingFace authentication fails"""
    def __init__(self, message="HuggingFace authentication failed"):
        super().__init__(message)

class DeviceError(FluxBaseException):
    """Raised when device configuration fails"""
    def __init__(self, device, message=None):
        self.device = device
        if message is None:
            message = f"Device '{device}' is not available or compatible"
        super().__init__(message)

class InsufficientMemoryError(FluxBaseException):
    """Raised when there's insufficient memory for generation"""
    def __init__(self, required_gb=None, available_gb=None):
        self.required_gb = required_gb
        self.available_gb = available_gb
        message = "Insufficient memory for image generation"
        if required_gb and available_gb:
            message += f" (required: {required_gb}GB, available: {available_gb}GB)"
        super().__init__(message)

class GenerationError(FluxBaseException):
    """Raised when image generation fails"""
    def __init__(self, message, stage=None):
        self.stage = stage
        if stage:
            message = f"Generation failed at {stage}: {message}"
        super().__init__(message)

class InvalidParameterError(FluxBaseException):
    """Raised when invalid parameters are provided"""
    def __init__(self, parameter, value, valid_range=None):
        self.parameter = parameter
        self.value = value
        self.valid_range = valid_range
        message = f"Invalid value for {parameter}: {value}"
        if valid_range:
            message += f" (valid range: {valid_range})"
        super().__init__(message)

class WebUIError(FluxBaseException):
    """Raised when web UI encounters errors"""
    def __init__(self, message, error_type="general"):
        self.error_type = error_type
        super().__init__(message)

def handle_common_errors(func):
    """Decorator for common error handling patterns"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            raise InsufficientMemoryError("GPU out of memory. Try using CPU or reducing image size.")
        except FileNotFoundError as e:
            if "models" in str(e):
                raise ModelNotFoundError("Required model files not found", str(e))
            raise
        except ImportError as e:
            raise FluxBaseException(f"Missing dependency: {e}")
        except Exception as e:
            # Re-raise known FLUX exceptions
            if isinstance(e, FluxBaseException):
                raise
            # Wrap unknown exceptions
            raise FluxBaseException(f"Unexpected error: {e}")
    return wrapper