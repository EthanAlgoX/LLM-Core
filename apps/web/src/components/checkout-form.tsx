export function CheckoutForm({ disabled }: { disabled?: boolean }) {
  return (
    <form action="/api/checkout" method="post">
      <button
        type="submit"
        disabled={disabled}
        className="inline-flex min-h-11 items-center justify-center rounded-full bg-slate-950 px-5 py-3 text-sm font-semibold text-white hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-teal-500 dark:text-slate-950 dark:hover:bg-teal-400"
      >
        {disabled ? "Configure Stripe first" : "Start subscription"}
      </button>
    </form>
  );
}
