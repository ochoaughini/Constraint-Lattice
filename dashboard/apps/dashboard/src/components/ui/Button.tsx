import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  className?: string;
}

export const Button: React.FC<ButtonProps> = ({ children, className = '', ...rest }) => (
  <button
    className={`px-4 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 focus:outline-none focus:ring ${className}`}
    {...rest}
  >
    {children}
  </button>
);

export default Button;
