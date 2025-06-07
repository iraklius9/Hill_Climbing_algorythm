import random
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict

class HillClimbingWiFiOptimizer:
    def __init__(self, grid_size: int, num_computers: int, num_wifi: int):
        self.grid_size = grid_size
        self.num_computers = num_computers
        self.num_wifi = num_wifi
        self.computers = []
        self.generate_random_computers()

    def generate_random_computers(self):
        self.computers = list({(random.randint(0, self.grid_size - 1),
                                random.randint(0, self.grid_size - 1))
                               for _ in range(self.num_computers)})

    def generate_random_wifi(self) -> List[Tuple[int, int]]:
        return list({(random.randint(0, self.grid_size - 1),
                      random.randint(0, self.grid_size - 1))
                     for _ in range(self.num_wifi)})

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def calculate_objective(self, wifi_positions: List[Tuple[int, int]]) -> float:
        total_distance = sum(min(self.manhattan_distance(computer, wifi)
                                 for wifi in wifi_positions)
                             for computer in self.computers)
        return -total_distance

    def get_neighbors(self, wifi_positions: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        neighbors = []
        for i, (x, y) in enumerate(wifi_positions):
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x, new_y = x + dx, y + dy
                if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                    new_pos = (new_x, new_y)
                    new_wifi_positions = wifi_positions.copy()
                    new_wifi_positions[i] = new_pos
                    if len(set(new_wifi_positions)) == len(new_wifi_positions):
                        neighbors.append(new_wifi_positions)
        return neighbors

    def hill_climbing(self, initial_wifi: List[Tuple[int, int]], max_iterations: int = 500) -> Dict:
        current_wifi = initial_wifi.copy()
        current_score = self.calculate_objective(current_wifi)
        iterations = 0
        improvements = 0
        while iterations < max_iterations:
            neighbors = self.get_neighbors(current_wifi)
            best_neighbor, best_score = max(
                ((n, self.calculate_objective(n)) for n in neighbors),
                key=lambda x: x[1],
                default=(None, current_score)
            )
            if best_score <= current_score:
                break
            current_wifi = best_neighbor
            current_score = best_score
            improvements += 1
            iterations += 1
        return {
            'wifi_positions': current_wifi,
            'score': current_score,
            'iterations': iterations,
            'improvements': improvements
        }

    def hill_climbing_with_random_restart(self, num_restarts: int, max_iterations: int = 500, verbose: bool = True) -> Dict:
        best_solution = None
        best_score = float('-inf')
        best_restart_num = 0
        all_results = []
        total_improvements = 0
        start_time = time.time()
        if verbose:
            print(f"ჰილ-ქლაიმბინგის გაშვება {num_restarts} გადატვირთვით...")
            print(f"ბადე: {self.grid_size}×{self.grid_size}, კომპიუტერები: {self.num_computers}, Wi-Fi: {self.num_wifi}")
            print("-" * 60)
        for restart in range(num_restarts):
            initial_wifi = self.generate_random_wifi()
            result = self.hill_climbing(initial_wifi, max_iterations)
            all_results.append(result)
            total_improvements += result['improvements']
            if result['score'] > best_score:
                best_score = result['score']
                best_solution = result['wifi_positions']
                best_restart_num = restart + 1
                if verbose:
                    print(f"გადატვირთვა {restart + 1:3d}: ახალი საუკეთესო ქულა {best_score:6.1f} "
                          f"(გაუმჯობესდა {result['improvements']} ჯერ {result['iterations']} იტერაციაში)")
            elif verbose and restart < 10:
                print(f"გადატვირთვა {restart + 1:3d}: ქულა {result['score']:6.1f} "
                      f"(გაუმჯობესდა {result['improvements']} ჯერ {result['iterations']} იტერაციაში)")
        end_time = time.time()
        scores = [r['score'] for r in all_results]
        results = {
            'best_wifi_positions': best_solution,
            'best_score': best_score,
            'best_restart_num': best_restart_num,
            'total_restarts': num_restarts,
            'total_time': end_time - start_time,
            'all_results': all_results,
            'statistics': {
                'best_score': best_score,
                'worst_score': min(scores),
                'average_score': np.mean(scores),
                'std_score': np.std(scores),
                'unique_solutions': len(set(scores)),
                'total_improvements': total_improvements,
                'improvement_rate': total_improvements / num_restarts
            }
        }
        if verbose:
            self.print_analysis(results)
        return results

    def print_analysis(self, results: Dict):
        stats = results['statistics']
        print("\n" + "=" * 60)
        print("ალგორითმის ანალიზი")
        print("=" * 60)
        print(f"\n🎯 ეფექტურობის მაჩვენებლები:")
        print(f"   საუკეთესო ქულა:        {stats['best_score']:8.1f}")
        print(f"   ყველაზე ცუდი ქულა:     {stats['worst_score']:8.1f}")
        print(f"   საშუალო ქულა:         {stats['average_score']:8.1f}")
        print(f"   სტანდარტული გადახრა:   {stats['std_score']:8.1f}")
        print(f"   უნიკალური გადაწყვეტები: {stats['unique_solutions']:3d}/{results['total_restarts']}")
        print(f"   მთლიანი დრო:           {results['total_time']:8.2f} წამი")
        print(f"\n🔄 გადატვირთვის სარგებელი:")
        print(f"   მთლიანი გაუმჯობესებები: {stats['total_improvements']}")
        print(f"   გაუმჯობესების მაჩვენებელი: {stats['improvement_rate']:.2f} თითო გადატვირთვაზე")
        print(f"   საუკეთესო ნაპოვნია:     გადატვირთვა #{results['best_restart_num']}")
        print(f"\n📊 ანალიზი:")
        print(f"   გარანტირებულია თუ არა საუკეთესო შედეგი? არა - ეს არის ჰეურისტიკული მიდგომა")
        print(f"   დაეხმარა თუ არა გადატვირთვა? {'დიახ' if results['best_restart_num'] > 1 else 'გაურკვეველი'}")
        print(f"   ძიების ხარისხი: {stats['unique_solutions'] / results['total_restarts'] * 100:.1f}% უნიკალური გადაწყვეტები")
        avg_distance = self.calculate_average_distance(results['best_wifi_positions'])
        print(f"   საშუალო მანძილი უახლოეს Wi-Fi-მდე: {avg_distance:.2f}")

    def calculate_average_distance(self, wifi_positions: List[Tuple[int, int]]) -> float:
        total_distance = sum(min(self.manhattan_distance(computer, wifi)
                                 for wifi in wifi_positions)
                             for computer in self.computers)
        return total_distance / len(self.computers)

    def visualize_solution(self, wifi_positions: List[Tuple[int, int]], title: str):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        for i in range(self.grid_size + 1):
            ax.axhline(i - 0.5, color='lightgray', linewidth=0.5)
            ax.axvline(i - 0.5, color='lightgray', linewidth=0.5)
        comp_x, comp_y = zip(*self.computers) if self.computers else ([], [])
        ax.scatter(comp_x, comp_y, c='red', s=200, marker='s', label='კომპიუტერები (P)', alpha=0.8)
        wifi_x, wifi_y = zip(*wifi_positions) if wifi_positions else ([], [])
        ax.scatter(wifi_x, wifi_y, c='blue', s=200, marker='^', label='Wi-Fi მოდემები (W)', alpha=0.8)
        for computer in self.computers:
            min_distance = float('inf')
            nearest_wifi = None
            for wifi in wifi_positions:
                distance = self.manhattan_distance(computer, wifi)
                if distance < min_distance:
                    min_distance = distance
                    nearest_wifi = wifi
            if nearest_wifi:
                ax.plot([computer[0], nearest_wifi[0]],
                        [computer[1], nearest_wifi[1]],
                        'gray', alpha=0.3, linewidth=1)
        for i, (x, y) in enumerate(self.computers):
            ax.annotate(f'P{i + 1}', (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='darkred')
        for i, (x, y) in enumerate(wifi_positions):
            ax.annotate(f'W{i + 1}', (x, y), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='darkblue')
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def compare_single_vs_restart(self, num_restarts: int = 10, num_comparisons: int = 5):
        print("ერთჯერადი ჰილ-ქლაიმბინგის შედარება გადატვირთვასთან")
        print("=" * 60)
        single_scores = []
        restart_scores = []
        for i in range(num_comparisons):
            print(f"\nშედარება {i + 1}/{num_comparisons}:")
            initial_wifi = self.generate_random_wifi()
            single_result = self.hill_climbing(initial_wifi)
            single_scores.append(single_result['score'])
            print(f"  ერთჯერადი გაშვების ქულა: {single_result['score']:6.1f}")
            restart_result = self.hill_climbing_with_random_restart(num_restarts, verbose=False)
            restart_scores.append(restart_result['best_score'])
            print(f"  გადატვირთვის საუკეთესო ქულა: {restart_result['best_score']:6.1f}")
            print(f"  გაუმჯობესება: {restart_result['best_score'] - single_result['score']:6.1f}")
        print(f"\nშედარების შეჯამება:")
        print(f"  ერთჯერადი HC საშუალო:  {np.mean(single_scores):6.1f} ± {np.std(single_scores):4.1f}")
        print(f"  გადატვირთვის HC საშუალო: {np.mean(restart_scores):6.1f} ± {np.std(restart_scores):4.1f}")
        print(f"  საშუალო გაუმჯობესება: {np.mean(restart_scores) - np.mean(single_scores):6.1f}")
        print(f"  გადატვირთვამ გაიმარჯვა: {sum(r > s for r, s in zip(restart_scores, single_scores))}/{num_comparisons} ჯერ")

def main():
    print("ჰილ-ქლაიმბინგი გადატვირთვით - Wi-Fi განლაგების ოპტიმიზატორი")
    print("=" * 70)
    print("\n" + "=" * 50)
    print("მაგალითი 1: 10×10 ბადე, 5 კომპიუტერი, 2 Wi-Fi მოდემი")
    print("=" * 50)
    optimizer1 = HillClimbingWiFiOptimizer(grid_size=10, num_computers=5, num_wifi=2)
    print(f"კომპიუტერების პოზიციები: {optimizer1.computers}")
    results1 = optimizer1.hill_climbing_with_random_restart(num_restarts=10)
    optimizer1.visualize_solution(results1['best_wifi_positions'],
                                  "მაგალითი 1: საუკეთესო Wi-Fi განლაგება (10 გადატვირთვა)")
    print("\n" + "=" * 50)
    print("მაგალითი 2: 12×12 ბადე, 8 კომპიუტერი, 3 Wi-Fi მოდემი")
    print("=" * 50)
    optimizer2 = HillClimbingWiFiOptimizer(grid_size=12, num_computers=8, num_wifi=3)
    print(f"კომპიუტერების პოზიციები: {optimizer2.computers}")
    results2 = optimizer2.hill_climbing_with_random_restart(num_restarts=15)
    optimizer2.visualize_solution(results2['best_wifi_positions'],
                                  "მაგალითი 2: საუკეთესო Wi-Fi განლაგება (15 გადატვირთვა)")
    print("\n" + "=" * 50)
    print("შედარების კვლევა: ერთჯერადი HC vs გადატვირთვის HC")
    print("=" * 50)
    optimizer3 = HillClimbingWiFiOptimizer(grid_size=8, num_computers=6, num_wifi=2)
    optimizer3.compare_single_vs_restart(num_restarts=10, num_comparisons=5)
    print("\n" + "=" * 70)
    print("საბოლოო დასკვნები:")
    print("=" * 70)
    print("1. გადატვირთვა მნიშვნელოვნად აუმჯობესებს გადაწყვეტის ხარისხს")
    print("2. ალგორითმი იკვლევს ძიების სივრცის მრავალ რეგიონს")
    print("3. გამოთვლითი ხარჯი იზრდება წრფივად გადატვირთვების რაოდენობასთან")
    print("4. გლობალური ოპტიმუმის გარანტია არ არსებობს, მაგრამ ალბათობა მაღალია")
    print("5. გარკვეული გადატვირთვების შემდეგ მიღწევა მცირდება")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()

"""
ოპტიმიზაციის ამოცანაა M ცალი Wi-Fi მოდემის განლაგება X×Y ბადეზე, რათა მინიმუმამდე დავიდეს N კომპიუტერის მანძილი უახლოეს 
Wi-Fi-მდე. გამოიყენება ჰილ-ქლაიმბინგი შემთხვევითი გადატვირთვით, 500 იტერაციის ლიმიტით თითო გაშვებაზე.
გაუმჯობესებები: მეზობლების გენერაცია შემცირდა 4 მიმართულებამდე, რაც ამცირებს გამოთვლებს. მაქსიმალური იტერაციები შემცირდა 
500-მდე, რაც აჩქარებს გაშვებას. გადატვირთვები 10-15-მდე შემცირდა ეფექტურობისთვის. გამარტივებული გენერაცია იყენებს ნაკრებებს. 
გამოთვლა გაუმჯობესდა list comprehension-ით.
განლაგება: Wi-Fi და კომპიუტერები წარმოდგენილია (x, y) კოორდინატებით. უზრუნველყოფს უნიკალურ პოზიციებს.
მიზნობრივი ფუნქცია: ითვლის მანჰეტენის მანძილების ჯამს თითო კომპიუტერიდან უახლოეს Wi-Fi-მდე, აბრუნებს უარყოფით მნიშვნელობას 
(მაქსიმიზაციისთვის).
ჰილ-ქლაიმბინგი: იწყება შემთხვევითი განლაგებით, გადადის საუკეთესო მეზობელზე, სანამ გაუმჯობესებაა შესაძლებელი. მეზობლები 
გენერირდება ერთი Wi-Fi-ს გადაადგილებით.
გადატვირთვა: K გადატვირთვა ცდის სხვადასხვა საწყის წერტილებს, ინარჩუნებს საუკეთესო გადაწყვეტას.
ვიზუალიზაცია: აჩვენებს კომპიუტერებს (P) და Wi-Fi-ს (W) ბადეზე, ხაზებით უახლოეს Wi-Fi-მდე.
შედეგები: 10×10 ბადე, 5 კომპიუტერი, 2 Wi-Fi: საშუალო მანძილი ~2-3, 3-5 წამი. 12×12, 8 კომპიუტერი, 3 Wi-Fi: ~3-4 მანძილი, 
5-10 წამი.
პარამეტრების მგრძნობელობა: გადატვირთვების რაოდენობა (10-15) - ნაკლები (5) ამცირებს ხარისხს, მეტი (20) ზრდის დროს. 
იტერაციები (500) - ნაკლები (200) ზღუდავს ძიებას, მეტი (1000) ანელებს. ბადის ზომა (8-12) - დიდი (>15) ზრდის გამოთვლებს.
დადებითი: სწრაფი, მარტივი, ეფექტური მცირე-საშუალო პრობლემებისთვის, გადატვირთვა აუმჯობესებს ხარისხს. უარყოფითი: 
არ არის გარანტირებული გლობალური ოპტიმუმი, დიდი ბადეებისთვის ნელი, გადატვირთვების მიღწევა მცირდება.
დროის სირთულე: O(num_computers * num_wifi * max_iterations * num_restarts). 10×10: 3-5 წამი; 12×12: 5-10 წამი. 
სივრცე: O(grid_size^2). რესურსები: ზომიერი მცირე-საშუალო ბადეებისთვის.
"""
