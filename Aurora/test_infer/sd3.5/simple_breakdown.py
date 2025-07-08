from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
import torch
import intel_extension_for_pytorch as ipex
import time
import numpy as np
from contextlib import contextmanager

class CorrectedPipelineProfiler:
    """Corrected profiler for SD3 pipeline with accurate time tracking"""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.profile_data = {}
        self.profiling_enabled = False
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Store original methods for later restoration"""
        self._original_methods = {
            'transformer_forward': self.pipeline.transformer.forward,
            'vae_decode': self.pipeline.vae.decode if hasattr(self.pipeline.vae, 'decode') else None,
            'vae_encode': self.pipeline.vae.encode if hasattr(self.pipeline.vae, 'encode') else None,
            'scheduler_step': self.pipeline.scheduler.step,
            'scheduler_set_timesteps': self.pipeline.scheduler.set_timesteps,
            'pipeline_call': self.pipeline.__call__,
        }
    
    @contextmanager
    def profile_module(self, module_name: str):
        """Context manager for profiling individual modules"""
        if not self.profiling_enabled:
            yield
            return
            
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            
            if module_name not in self.profile_data:
                self.profile_data[module_name] = []
            self.profile_data[module_name].append(elapsed)
    
    def profile_inference(self, *args, **kwargs):
        """Run inference with comprehensive profiling"""
        self.profiling_enabled = True
        
        # Create wrapped methods
        def profiled_transformer_forward(*t_args, **t_kwargs):
            with self.profile_module("transformer"):
                return self._original_methods['transformer_forward'](*t_args, **t_kwargs)
        
        def profiled_vae_decode(*v_args, **v_kwargs):
            with self.profile_module("vae_decode"):
                return self._original_methods['vae_decode'](*v_args, **v_kwargs)
        
        def profiled_vae_encode(*v_args, **v_kwargs):
            with self.profile_module("vae_encode"):
                return self._original_methods['vae_encode'](*v_args, **v_kwargs)
        
        def profiled_scheduler_step(*s_args, **s_kwargs):
            with self.profile_module("scheduler_step"):
                return self._original_methods['scheduler_step'](*s_args, **s_kwargs)
        
        def profiled_scheduler_set_timesteps(*s_args, **s_kwargs):
            with self.profile_module("scheduler_setup"):
                return self._original_methods['scheduler_set_timesteps'](*s_args, **s_kwargs)
        
        # Override the main call to track other operations
        original_call = self._original_methods['pipeline_call']
        
        def profiled_call(*args, **kwargs):
            # Track total pipeline time
            pipeline_start = time.perf_counter()
            
            # Mark the start of this run to track only operations from this call
            run_start_counts = {k: len(v) for k, v in self.profile_data.items()}
            
            with self.profile_module("total_pipeline"):
                result = original_call(*args, **kwargs)
            
            pipeline_end = time.perf_counter()
            total_time = pipeline_end - pipeline_start
            
            # Calculate time for operations in this run only
            tracked_time = 0
            for module, times in self.profile_data.items():
                if module not in ["total_pipeline", "other_operations"]:
                    # Only count times added during this run
                    start_idx = run_start_counts.get(module, 0)
                    tracked_time += sum(times[start_idx:])
            
            # Record untracked time (should be positive)
            other_time = total_time - tracked_time
            if "other_operations" not in self.profile_data:
                self.profile_data["other_operations"] = []
            self.profile_data["other_operations"].append(max(0, other_time))
            
            return result
        
        # Replace methods
        self.pipeline.transformer.forward = profiled_transformer_forward
        if self._original_methods['vae_decode']:
            self.pipeline.vae.decode = profiled_vae_decode
        if self._original_methods['vae_encode']:
            self.pipeline.vae.encode = profiled_vae_encode
        self.pipeline.scheduler.step = profiled_scheduler_step
        self.pipeline.scheduler.set_timesteps = profiled_scheduler_set_timesteps
        
        try:
            result = profiled_call(*args, **kwargs)
            return result
        finally:
            # Restore original methods
            self.pipeline.transformer.forward = self._original_methods['transformer_forward']
            if self._original_methods['vae_decode']:
                self.pipeline.vae.decode = self._original_methods['vae_decode']
            if self._original_methods['vae_encode']:
                self.pipeline.vae.encode = self._original_methods['vae_encode']
            self.pipeline.scheduler.step = self._original_methods['scheduler_step']
            self.pipeline.scheduler.set_timesteps = self._original_methods['scheduler_set_timesteps']
            self.profiling_enabled = False
    
    def get_profile_summary(self):
        """Get corrected profiling summary statistics"""
        results = {}
        
        # Calculate totals for each module
        for module_name, times in self.profile_data.items():
            if times:
                results[module_name] = {
                    'mean_per_call': np.mean(times),
                    'total': np.sum(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times)
                }
        
        # Calculate percentages based on total pipeline time
        if 'total_pipeline' in results:
            # Use the mean of total_pipeline calls for percentage calculation
            total_pipeline_mean = results['total_pipeline']['mean_per_call']
            
            for module_name in results:
                if module_name != 'total_pipeline':
                    # For modules called multiple times per pipeline run, use their total time per pipeline run
                    module_total_per_run = results[module_name]['total'] / results['total_pipeline']['count']
                    results[module_name]['percentage'] = (module_total_per_run / total_pipeline_mean) * 100
                else:
                    results[module_name]['percentage'] = 100.0
        
        return results
    
    def print_profile_report(self):
        """Print corrected profiling report"""
        summary = self.get_profile_summary()
        
        print(f"\n{'='*85}")
        print(f"CORRECTED PIPELINE PROFILING REPORT")
        print(f"{'='*85}")
        print(f"{'Module':<20} {'Total (s)':<12} {'Calls':<8} {'Mean/Call':<12} {'% of Pipeline':<14}")
        print(f"{'-'*85}")
        
        # Sort by percentage descending
        sorted_modules = sorted(summary.items(), 
                               key=lambda x: x[1].get('percentage', 0), 
                               reverse=True)
        
        for module_name, stats in sorted_modules:
            print(f"{module_name:<20} {stats['total']:<12.4f} {stats['count']:<8} "
                  f"{stats['mean_per_call']:<12.4f} {stats['percentage']:<14.1f}")
        
        print(f"{'='*85}")
        
        # Verification
        if 'total_pipeline' in summary:
            pipeline_runs = summary['total_pipeline']['count']
            print(f"\nPipeline runs: {pipeline_runs}")
            print(f"Average pipeline time: {summary['total_pipeline']['mean_per_call']:.2f}s")
            
            # Check time accounting per run
            total_per_run = summary['total_pipeline']['mean_per_call']
            accounted_per_run = sum(
                stats['total'] / pipeline_runs 
                for name, stats in summary.items() 
                if name not in ['total_pipeline', 'other_operations']
            )
            other_per_run = summary.get('other_operations', {}).get('total', 0) / pipeline_runs
            
            print(f"\nTime breakdown per pipeline run:")
            print(f"  Total: {total_per_run:.2f}s")
            print(f"  Tracked modules: {accounted_per_run:.2f}s ({accounted_per_run/total_per_run*100:.1f}%)")
            print(f"  Other operations: {other_per_run:.2f}s ({other_per_run/total_per_run*100:.1f}%)")
        
        # Detailed breakdown
        if 'transformer' in summary:
            trans_stats = summary['transformer']
            calls_per_run = trans_stats['count'] / summary['total_pipeline']['count']
            print(f"\nTransformer Details:")
            print(f"  - Calls per pipeline run: {calls_per_run:.0f}")
            print(f"  - Time per call: {trans_stats['mean_per_call']:.4f}s")
            print(f"  - Total time per run: {trans_stats['total']/summary['total_pipeline']['count']:.2f}s")


def profile_denoising_loop(pipeline, device='cpu', num_steps=20):
    """Profile the denoising loop in detail"""
    print(f"\n{'='*70}")
    print("DENOISING LOOP PROFILING")
    print(f"{'='*70}")
    
    # Setup
    batch_size = 1
    height, width = 1024, 1024
    latent_channels = pipeline.transformer.config.in_channels
    
    # Create inputs
    latents = torch.randn(
        batch_size, latent_channels, height // 8, width // 8,
        device=device, dtype=torch.bfloat16
    )
    
    zero_embeds, zero_pooled = get_zero_embeddings(pipeline, batch_size=1, device=device)
    
    # Set timesteps
    pipeline.scheduler.set_timesteps(num_steps, device=device)
    
    # Profile a few denoising steps
    step_times = []
    
    print(f"\nProfiling first 5 and last 5 steps of {num_steps} total steps:")
    
    for i, t in enumerate(pipeline.scheduler.timesteps[:10]):  # First 10 steps
        if i >= 5 and i < num_steps - 5:
            continue  # Skip middle steps
            
        step_start = time.perf_counter()
        
        # 1. Transformer
        trans_start = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            noise_pred = pipeline.transformer(
                hidden_states=latents,
                timestep=t.unsqueeze(0) if t.dim() == 0 else t,
                encoder_hidden_states=zero_embeds,
                pooled_projections=zero_pooled,
                return_dict=False,
            )[0]
        trans_time = time.perf_counter() - trans_start
        
        # 2. Scheduler step
        sched_start = time.perf_counter()
        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        sched_time = time.perf_counter() - sched_start
        
        step_time = time.perf_counter() - step_start
        step_times.append({
            'step': i,
            'total': step_time,
            'transformer': trans_time,
            'scheduler': sched_time,
            'other': step_time - trans_time - sched_time
        })
        
        print(f"Step {i:3d}: Total={step_time*1000:6.1f}ms "
              f"(Trans={trans_time*1000:6.1f}ms, Sched={sched_time*1000:5.2f}ms, "
              f"Other={(step_time-trans_time-sched_time)*1000:5.2f}ms)")
    
    # Summary
    avg_total = np.mean([s['total'] for s in step_times])
    avg_trans = np.mean([s['transformer'] for s in step_times])
    avg_sched = np.mean([s['scheduler'] for s in step_times])
    avg_other = np.mean([s['other'] for s in step_times])
    
    print(f"\nAverage per step:")
    print(f"  Total: {avg_total*1000:.1f}ms")
    print(f"  Transformer: {avg_trans*1000:.1f}ms ({avg_trans/avg_total*100:.1f}%)")
    print(f"  Scheduler: {avg_sched*1000:.2f}ms ({avg_sched/avg_total*100:.1f}%)")
    print(f"  Other: {avg_other*1000:.2f}ms ({avg_other/avg_total*100:.1f}%)")
    
    print(f"\nEstimated total time for {num_steps} steps: {avg_total*num_steps:.2f}s")


def get_zero_embeddings(pipeline, batch_size=1, device=None):
    """Helper to get zero embeddings"""
    if device is None:
        device = pipeline._execution_device
        
    with torch.no_grad():
        encode_result = pipeline.encode_prompt(
            prompt="",
            prompt_2="",
            prompt_3="",
            device=device,
            num_images_per_prompt=batch_size,
            do_classifier_free_guidance=False,
        )
        
        if isinstance(encode_result, tuple) and len(encode_result) == 2:
            prompt_embeds, pooled_embeds = encode_result
        elif isinstance(encode_result, tuple) and len(encode_result) >= 3:
            prompt_embeds, _, pooled_embeds = encode_result[:3]
        else:
            raise ValueError(f"Unexpected return format from encode_prompt")
    
    zero_prompt_embeds = torch.zeros_like(prompt_embeds)
    zero_pooled_embeds = torch.zeros_like(pooled_embeds)
    
    return zero_prompt_embeds, zero_pooled_embeds


def main():
    """Main execution with corrected profiling"""
    from diffusers import StableDiffusion3Pipeline
    
    # Configuration
    device = 'xpu'  # or 'cpu'
    model_id = "stabilityai/stable-diffusion-3.5-medium"
    num_runs = 3
    num_steps = 20
    
    print("Loading SD3.5 model...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )
    
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )
    
    pipeline = pipeline.to(device)
    
    print("Applying IPEX optimizations...")
    pipeline.transformer.eval()
    pipeline.transformer = ipex.optimize(pipeline.transformer, dtype=torch.bfloat16, level="O1")
    
    if hasattr(pipeline, 'vae'):
        pipeline.vae.eval()
        pipeline.vae = ipex.optimize(pipeline.vae, dtype=torch.bfloat16)
    
    # Create profiler
    profiler = CorrectedPipelineProfiler(pipeline)
    
    # Get embeddings
    zero_embeds, zero_pooled = get_zero_embeddings(pipeline, batch_size=1, device=device)
    
    print(f"\nRunning corrected profiling ({num_runs} runs, {num_steps} steps)...")
    
    # Warmup
    print("Warmup...")
    with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
        _ = pipeline(
            prompt_embeds=zero_embeds.clone(),
            pooled_prompt_embeds=zero_pooled.clone(),
            num_inference_steps=5,
            guidance_scale=1.0,
        ).images[0]
    
    # Clear any existing profile data
    profiler.profile_data.clear()
    
    # Profile runs
    for i in range(num_runs):
        print(f"Profiling run {i+1}/{num_runs}...")
        
        with torch.no_grad(), torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            _ = profiler.profile_inference(
                prompt_embeds=zero_embeds.clone(),
                pooled_prompt_embeds=zero_pooled.clone(),
                num_inference_steps=num_steps,
                guidance_scale=1.0,
            ).images[0]
    
    # Print results
    profiler.print_profile_report()
    
    # Detailed denoising loop analysis
    profile_denoising_loop(pipeline, device=device, num_steps=num_steps)
    
    print("\nCorrected profiling complete!")


if __name__ == "__main__":
    main()