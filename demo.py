import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# PART 1: EHO ALGORITHM (Main Implementation)
# ============================================================================

class ElephantHerdingOptimization:
    """Elephant Herding Optimization Algorithm"""
    
    def __init__(self, n_elephants=50, n_clans=5, alpha=0.5, beta=0.1, 
                 max_iter=100, dim=2, bounds=(-10, 10)):
        self.n_elephants = n_elephants
        self.n_clans = n_clans
        self.elephants_per_clan = n_elephants // n_clans
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.dim = dim
        self.bounds = bounds
        
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_curve = []
        
    def initialize_population(self):
        """Initialize elephant positions randomly"""
        lower, upper = self.bounds
        self.population = np.random.uniform(lower, upper, (self.n_elephants, self.dim))
        
    def evaluate_fitness(self, objective_function):
        """Evaluate fitness of all elephants"""
        self.fitness = np.array([objective_function(ind) for ind in self.population])
        
        min_idx = np.argmin(self.fitness)
        if self.fitness[min_idx] < self.best_fitness:
            self.best_fitness = self.fitness[min_idx]
            self.best_solution = self.population[min_idx].copy()
    
    def clan_updating_operator(self, clan_idx):
        """Update positions based on matriarch"""
        start_idx = clan_idx * self.elephants_per_clan
        end_idx = start_idx + self.elephants_per_clan
        
        clan_elephants = self.population[start_idx:end_idx]
        clan_fitness = self.fitness[start_idx:end_idx]
        
        matriarch_idx = np.argmin(clan_fitness)
        matriarch = clan_elephants[matriarch_idx]
        clan_center = np.mean(clan_elephants, axis=0)
        
        for i in range(self.elephants_per_clan):
            if i != matriarch_idx:
                r = np.random.rand()
                new_position = clan_elephants[i] + self.alpha * (matriarch - clan_elephants[i]) * r
                new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
                self.population[start_idx + i] = new_position
            else:
                new_position = self.beta * clan_center
                new_position = np.clip(new_position, self.bounds[0], self.bounds[1])
                self.population[start_idx + i] = new_position
    
    def separating_operator(self):
        """Replace worst elephant in each clan"""
        for clan_idx in range(self.n_clans):
            start_idx = clan_idx * self.elephants_per_clan
            end_idx = start_idx + self.elephants_per_clan
            
            clan_fitness = self.fitness[start_idx:end_idx]
            worst_idx = start_idx + np.argmax(clan_fitness)
            
            lower, upper = self.bounds
            new_position = lower + (upper - lower) * np.random.rand(self.dim)
            self.population[worst_idx] = new_position
    
    def optimize(self, objective_function, verbose=True):
        """Main optimization loop"""
        self.initialize_population()
        self.convergence_curve = []
        
        for iteration in range(self.max_iter):
            self.evaluate_fitness(objective_function)
            self.convergence_curve.append(self.best_fitness)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Best Fitness = {self.best_fitness:.8f}")
            
            for clan_idx in range(self.n_clans):
                self.clan_updating_operator(clan_idx)
            
            self.separating_operator()
        
        self.evaluate_fitness(objective_function)
        self.convergence_curve.append(self.best_fitness)
        
        if verbose:
            print(f"\n✅ Optimization Complete!")
            print(f"Best Fitness: {self.best_fitness:.10f}")
            print(f"Best Solution: {self.best_solution}")
        
        return self.best_solution, self.best_fitness, self.convergence_curve


# ============================================================================
# PART 2: TEST FUNCTIONS
# ============================================================================

def sphere_function(x):
    """Sphere: f(x) = sum(x²), Optimum: 0"""
    return np.sum(x**2)

def rastrigin_function(x):
    """Rastrigin: Multimodal, Optimum: 0"""
    n = len(x)
    return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

def rosenbrock_function(x):
    """Rosenbrock: Valley-shaped, Optimum: 0"""
    return np.sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def ackley_function(x):
    """Ackley: Multimodal, Optimum: 0"""
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2*np.pi*x))
    return -20*np.exp(-0.2*np.sqrt(sum1/n)) - np.exp(sum2/n) + 20 + np.e


# ============================================================================
# PART 3: VISUALIZATION
# ============================================================================

def plot_all_results(eho, function, function_name, bounds):
    """Create all visualizations in one figure"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Convergence Curve
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(eho.convergence_curve, 'b-', linewidth=2.5)
    ax1.set_xlabel('Iteration', fontweight='bold')
    ax1.set_ylabel('Best Fitness', fontweight='bold')
    ax1.set_title(f'Convergence Curve - {function_name}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Population Distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(eho.population[:, 0], eho.population[:, 1], 
                c='blue', alpha=0.6, s=80, label='Elephants')
    ax2.scatter(eho.best_solution[0], eho.best_solution[1], 
                c='red', marker='*', s=500, label='Best', zorder=5)
    ax2.set_xlim(bounds)
    ax2.set_ylim(bounds)
    ax2.set_xlabel('X₁', fontweight='bold')
    ax2.set_ylabel('X₂', fontweight='bold')
    ax2.set_title('Final Population Distribution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 3D Surface
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    x = np.linspace(bounds[0], bounds[1], 50)
    y = np.linspace(bounds[0], bounds[1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(50):
        for j in range(50):
            Z[i, j] = function(np.array([X[i, j], Y[i, j]]))
    
    ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax3.set_xlabel('X₁')
    ax3.set_ylabel('X₂')
    ax3.set_zlabel('f(X)')
    ax3.set_title('3D Function Landscape', fontweight='bold')
    
    # 4. Contour Plot
    ax4 = plt.subplot(2, 3, 4)
    contour = ax4.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
    ax4.contour(X, Y, Z, levels=15, colors='black', alpha=0.2, linewidths=0.5)
    ax4.scatter(eho.best_solution[0], eho.best_solution[1], 
                c='red', marker='*', s=400, edgecolors='white', linewidths=2)
    ax4.set_xlabel('X₁', fontweight='bold')
    ax4.set_ylabel('X₂', fontweight='bold')
    ax4.set_title('Contour Plot with Best Solution', fontweight='bold')
    plt.colorbar(contour, ax=ax4)
    
    # 5. Clan Distribution (colored by clan)
    ax5 = plt.subplot(2, 3, 5)
    elephants_per_clan = eho.n_elephants // eho.n_clans
    colors = plt.cm.Set3(np.linspace(0, 1, eho.n_clans))
    
    for clan_idx in range(eho.n_clans):
        start = clan_idx * elephants_per_clan
        end = start + elephants_per_clan
        clan_pop = eho.population[start:end]
        ax5.scatter(clan_pop[:, 0], clan_pop[:, 1], 
                   c=[colors[clan_idx]], s=100, alpha=0.7, 
                   label=f'Clan {clan_idx+1}')
    
    ax5.scatter(eho.best_solution[0], eho.best_solution[1], 
                c='red', marker='*', s=500, label='Best', zorder=10)
    ax5.set_xlim(bounds)
    ax5.set_ylim(bounds)
    ax5.set_xlabel('X₁', fontweight='bold')
    ax5.set_ylabel('X₂', fontweight='bold')
    ax5.set_title('Clan Distribution', fontweight='bold')
    ax5.legend(fontsize=8, ncol=2)
    ax5.grid(True, alpha=0.3)
    
    # 6. Statistics Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
    OPTIMIZATION RESULTS
    {'='*35}
    
    Function: {function_name}
    
    Best Fitness: {eho.best_fitness:.6e}
    Best Solution: [{eho.best_solution[0]:.4f}, {eho.best_solution[1]:.4f}]
    
    Parameters:
    • Elephants: {eho.n_elephants}
    • Clans: {eho.n_clans}
    • Alpha: {eho.alpha}
    • Beta: {eho.beta}
    • Iterations: {eho.max_iter}
    
    Final Stats:
    • Mean Fitness: {np.mean(eho.fitness):.6e}
    • Std Fitness: {np.std(eho.fitness):.6e}
    • Population Diversity: {np.std(eho.population):.4f}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f'eho_{function_name.lower()}_results.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# PART 4: MAIN DEMO (Run this for your video!)
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🐘 ELEPHANT HERDING OPTIMIZATION - COMPLETE DEMONSTRATION")
    print("="*70 + "\n")
    
    # Test functions to demonstrate
    test_functions = [
        (sphere_function, "Sphere Function", (-10, 10)),
        (rastrigin_function, "Rastrigin Function", (-5.12, 5.12)),
        (ackley_function, "Ackley Function", (-5, 5)),
    ]
    
    for func, name, bounds in test_functions:
        print(f"\n{'='*70}")
        print(f"📊 TESTING: {name}")
        print(f"{'='*70}")
        
        # Create and run optimizer
        eho = ElephantHerdingOptimization(
            n_elephants=50,
            n_clans=5,
            alpha=0.5,
            beta=0.1,
            max_iter=100,
            dim=2,
            bounds=bounds
        )
        
        print(f"\n🔧 Configuration:")
        print(f"   Population: {eho.n_elephants} elephants in {eho.n_clans} clans")
        print(f"   Parameters: α={eho.alpha}, β={eho.beta}")
        print(f"   Max Iterations: {eho.max_iter}")
        print(f"   Search Space: {bounds}")
        
        print(f"\n🚀 Starting Optimization...\n")
        
        best_sol, best_fit, curve = eho.optimize(func, verbose=True)
        
        # Create comprehensive visualization
        print(f"\n📊 Generating visualizations...")
        plot_all_results(eho, func, name, bounds)
        
        print(f"\n✅ {name} Complete!")
        print(f"   Image saved: eho_{name.lower().replace(' ', '_')}_results.png")
        print(f"   Best fitness: {best_fit:.10f}")
        
        input(f"\n⏸  Press Enter to continue to next function...")
    
    print("\n" + "="*70)
    print("🎉 ALL DEMONSTRATIONS COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("   - eho_sphere_function_results.png")
    print("   - eho_rastrigin_function_results.png")
    print("   - eho_ackley_function_results.png")
    print("\n✅ Use these images in your presentation!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()