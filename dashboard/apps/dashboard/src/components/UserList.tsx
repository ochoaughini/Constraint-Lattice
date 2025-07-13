import React from 'react';

interface User {
  id: string | number;
  firstName: string;
  lastName: string;
}

interface UserListProps {
  users: User[];
}

const UserList: React.FC<UserListProps> = ({ users }) => {
  const processedUsers = React.useMemo(
    () => users.map(u => ({ ...u, fullName: `${u.firstName} ${u.lastName}` })),
    [users]
  );

  return (
    <ul>
      {processedUsers.map(user => (
        <li key={user.id}>{user.fullName}</li>
      ))}
    </ul>
  );
};

export default UserList;
