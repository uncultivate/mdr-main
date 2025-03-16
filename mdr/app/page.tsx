import MagicSquareVisualizer from '@/components/MagicSquareVisualizer';

export default function Home() {
  return (
    <main className="min-h-screen p-8 bg-[#001525]">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold mb-8 text-[#24e4f2]">Magic Square Detection</h1>
        <p className="text-[#24e4f2] mb-8">Welcome to the Magic Square Detection game! Use arrow keys to navigate the grid and find magic squares.</p>
        <MagicSquareVisualizer />
      </div>
    </main>
  );
}